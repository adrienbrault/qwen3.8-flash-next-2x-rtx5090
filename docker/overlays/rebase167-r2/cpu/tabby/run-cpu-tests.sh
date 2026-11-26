#!/bin/bash
# TABBY rebase165 CPU tests. Run INSIDE the built image, no GPU, no network:
#   docker run --rm --network=none -v <W>/out/tabby:/tabby-tests:ro --entrypoint bash <image> /tabby-tests/run-cpu-tests.sh
# Env: TABBY_DIR (default /app), EXL3_TREE (default: the directory holding the installed exllamav3 package),
#      EXL3_STUB_EXT (default: 1 when `import exllamav3` fails without a GPU, else unset = the real extension),
#      PY (default python3).
# Suites: (1) test_tabby_callsites.py  TabbyAPI -> exllamav3 call sites + served contracts (cache_trace, R828
#         COUNTERS, loop-think, num_draft_tokens_by_batch, phase-switch methods); (2) test_tabby_rebase165.py  the
#         Tabby-side rebase logic (encode cache + offloop, recurrent_dtype, stream_generate surface, daily config);
#         (3) TabbyAPI's own unittest modules (served + upstream) that need neither pytest nor network.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-python3}"
export PYTHONDONTWRITEBYTECODE=1
export TABBY_DIR="${TABBY_DIR:-/app}"
if [ -z "${EXL3_TREE:-}" ]; then
  EXL3_TREE="$("$PY" -c 'import importlib.util as u, os; s = u.find_spec("exllamav3"); print(os.path.dirname(os.path.dirname(s.origin)))')"
fi
export EXL3_TREE
if [ -z "${EXL3_STUB_EXT:-}" ]; then
  if (cd /tmp && "$PY" -c 'import exllamav3' >/dev/null 2>&1); then EXL3_STUB_EXT=0; else EXL3_STUB_EXT=1; fi
fi
export EXL3_STUB_EXT
echo "python:   $PY ($("$PY" --version 2>&1))"
echo "tabby:    $TABBY_DIR ($(cat "$TABBY_DIR/TABBY.txt" 2>/dev/null | head -1))"
echo "exl3:     $EXL3_TREE (stub ext: $EXL3_STUB_EXT)"
echo
FAILED=0
run() {
  echo "----- $* -----"
  if "$@"; then echo "     [suite OK]"; else echo "     [suite FAILED]"; FAILED=1; fi
  echo
}
run "$PY" "$HERE/tests/test_tabby_callsites.py"
run "$PY" "$HERE/tests/test_tabby_rebase165.py"
run "$PY" "$HERE/tests/test_names.py"
# TabbyAPI's own suites. Not run here: test_disk_cache_config, test_generator_recovery, test_grammar_fail_closed,
# test_inline_model_loading, test_port_probe (need pytest, absent from the image); req_*.py (live-server clients).
TABBY_SUITES="test_tool_choice test_tool_choice_collector test_metrics_exposition test_metrics_record
  test_context_length_errors test_continue_final_message test_inline_model_identity
  test_model_list test_stream_parser test_stream_role test_template_vars test_timings test_tool_message_order
  test_usage_stats test_token_encode_endpoint test_web_ui test_tojson_filter"
for t in $TABBY_SUITES; do
  run "$PY" "$HERE/tests/_run_unittest.py" "tests.$t"
done
# The llguidance-backed grammar tests skip unless asked; the image has llguidance, so require them here
run env TABBY_TOOLCHOICE_REQUIRE_LLG=1 "$PY" "$HERE/tests/_run_unittest.py" tests.test_tool_choice
echo "===== ALL SUITES: $FAILED (0 = pass) ====="
exit "$FAILED"
