# r787-common.sh -- shared by r787a..d and r787-chain.sh (2026-09-27): the R785 daily's expected configuration, the
# per-boot provenance gate, the traffic quiesce (Hermes, Open WebUI's proxy) and a report-only foreign-request count.
# One copy instead of five: lib/serve-ctl.sh's header records how copied helpers drift (wrong queue names, wrong checks).
#
# Source AFTER /srv/qwen5090/lib/serve-ctl.sh (uses assert_env_keys) and after the caller's log() exists.
# Installed at /srv/qwen5090/r787-common.sh (operator; .new + mv).
#
# THE CONFIGURATION THE FIGURES MUST DESCRIBE (R785, promoted 2026-09-27 13:02 UTC; scripts/launch-flashnext.sh):
#   live launcher md5 e3db755f24a24192711e7edb96347d83, image tabbyapi:rebase-dev-r3, pool 901,120 @ 8,8, split [30, 30],
#   42 EXL3 env keys incl. EXL3_GR_MIX_TILED=1, TUNEDIR /srv/qwen5090/.exl3cache-rebase-dev-r3. Booted by the units with
#   env -i and NVME_TIER= (tier off, as every published curve). The launcher itself sets memory +4500, core 0 and stock
#   power at every boot (MEMOC / POWER blocks); the units only READ them and refuse anything else.
R787_MODEL=qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
R787_LIVE=/srv/qwen5090/launch-flashnext.sh
R787_MD5=${R787_MD5:-e3db755f24a24192711e7edb96347d83}
R787_IMG=${R787_IMG:-tabbyapi:rebase-dev-r3}
R787_POOL=${R787_POOL:-901120}
R787_KEYS=${R787_KEYS:-42}
R787_SPLIT=${R787_SPLIT:-"30, 30"}
R787_TUNEDIR=${R787_TUNEDIR:-/srv/qwen5090/.exl3cache-rebase-dev-r3}
# one date for the whole chain (r787-chain.sh exports it), so a chain that crosses midnight UTC keeps one set of dir names
R787_DATE=${R787_DATE:-$(date +%F)}
R787_WANT_GPC=${R787_WANT_GPC:-"0 0"}
R787_WANT_MEM=${R787_WANT_MEM:-"4500 4500"}
# Flash-Next publishes at stock (600 / 575 W): power.limit must equal power.default_limit on every card
R787_WANT_PWR=${R787_WANT_PWR:-$(nvidia-smi --query-gpu=power.default_limit --format=csv,noheader,nounits | awk '{printf "%s%.0f", (NR>1?" ":""), $1}')}
# Containers that send requests to :8022 WITHOUT going through Olla (so gateway_drain does not stop them):
#   hermes, hermes-webui  Hermes calls :8022 directly (r785 stopped both for its gates)
#   owui-proxy            Open WebUI's upstream picker; UPSTREAMS[0] is http://172.17.0.1:8022 (/srv/owui-proxy/proxy.py)
# Stopping owui-proxy leaves Open WebUI up with no models for the unit's / chain's duration. R787_QUIESCE="hermes
# hermes-webui" keeps Open WebUI on (its requests would then show in the foreign counts).
R787_QUIESCE=${R787_QUIESCE-"hermes hermes-webui owui-proxy"}
R787_STOPPED=

r787_gpc(){ timeout 30 sudo python3 -c 'import pynvml as N;N.nvmlInit();print(*[N.nvmlDeviceGetGpcClkVfOffset(N.nvmlDeviceGetHandleByIndex(i)) for i in range(N.nvmlDeviceGetCount())])' 2>/dev/null || echo "?"; }
r787_mem(){ timeout 30 sudo python3 -c 'import pynvml as N;N.nvmlInit();print(*[N.nvmlDeviceGetMemClkVfOffset(N.nvmlDeviceGetHandleByIndex(i)) for i in range(N.nvmlDeviceGetCount())])' 2>/dev/null || echo "?"; }
r787_pwr(){ nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits 2>/dev/null | awk '{printf "%s%.0f", (NR>1?" ":""), $1}'; }
r787_md5(){ md5sum < "$1" | cut -c1-32; }

# The live launcher is the R785 one. Anything else means the daily moved since this chain was written: the figures would
# describe another configuration, so the unit refuses to start (prints the reason, rc 1).
r787_launcher_ok(){ local m
  m=$(r787_md5 "$R787_LIVE" 2>/dev/null)
  [ "$m" = "$R787_MD5" ] || { echo "live launcher md5 '$m' != $R787_MD5 (the daily changed since R785)"; return 1; }
  [ "$(grep -oE '^DAILY_IMG=\S+' "$R787_LIVE" | cut -d= -f2)" = "$R787_IMG" ] || { echo "live DAILY_IMG is not $R787_IMG"; return 1; }
  return 0; }

# r787_boot_ok BOOTLOG -- after a boot of the live launcher (served id already confirmed by the caller). Prints one
# provenance line (image + id, env keys, pool, split, tunedir, tier, power, offsets, the launcher's own readback lines);
# appends " | BAD: <reasons>" and returns 1 when the served container is not the R785 daily at the published regime.
r787_boot_ok(){ local bl=$1 why= img imgid tier keys pwr gpc mem ml pl
  img=$(sudo docker inspect -f '{{.Config.Image}}' flashnext 2>/dev/null)
  imgid=$(sudo docker inspect -f '{{.Image}}' flashnext 2>/dev/null | cut -c1-19)
  tier=$(sudo docker exec flashnext env 2>/dev/null | grep -c '^EXL3_NVME_TIER=')
  keys=$(assert_env_keys "$bl" "$R787_KEYS" EXL3_GR_MIX_TILED 2>&1) || why="$why env keys: $keys;"
  [ "$img" = "$R787_IMG" ] || why="$why image '$img' != $R787_IMG;"
  grep -aqF "cache $R787_POOL @" "$bl" || why="$why pool $(grep -aoE 'cache [0-9]+ @' "$bl" | tail -1) != $R787_POOL;"
  grep -aqF "split [$R787_SPLIT]" "$bl" || why="$why $(grep -aoE 'split \[[^]]*\]' "$bl" | tail -1) != split [$R787_SPLIT];"
  grep -aqF "tunedir $R787_TUNEDIR," "$bl" || why="$why tunedir is not $R787_TUNEDIR;"
  [ "${tier:-0}" = 0 ] || why="$why NVMe tier on;"
  pwr=$(r787_pwr); gpc=$(r787_gpc); mem=$(r787_mem)
  [ "$pwr" = "$R787_WANT_PWR" ] || why="$why power '$pwr' W != stock '$R787_WANT_PWR';"
  [ "$gpc" = "$R787_WANT_GPC" ] || why="$why core offsets '$gpc' != '$R787_WANT_GPC';"
  [ "$mem" = "$R787_WANT_MEM" ] || why="$why memory offsets '$mem' != '$R787_WANT_MEM';"
  ml=$(grep -aoE 'memory clock offset: .*' "$bl" | tail -1); pl=$(grep -aoE 'power policy .*' "$bl" | tail -1)
  echo "image $img ($imgid); env keys $(grep -aoE 'env keys \([0-9]+\)' "$bl" | tail -1 | tr -dc 0-9) (tiled $(grep -acE 'env keys .*EXL3_GR_MIX_TILED=1' "$bl")); $(grep -aoE 'cache [0-9]+ @ [0-9]+,[0-9]+' "$bl" | tail -1); $(grep -aoE 'split \[[^]]*\]' "$bl" | tail -1); tier $tier; power $pwr W; core $gpc; memory $mem; launcher: ${ml:-no memory-offset line} / ${pl:-no power line}; $(grep -aoE 'VRAM free MiB [0-9/]+' "$bl" | tail -1)${why:+ | BAD:$why}"
  [ -z "$why" ]; }

# Clocks / power again, e.g. after a measurement: prints "power P W; core G; memory M" and rc 1 on a drift.
r787_clocks_ok(){ local pwr gpc mem
  pwr=$(r787_pwr); gpc=$(r787_gpc); mem=$(r787_mem)
  echo "power $pwr W; core $gpc; memory $mem"
  [ "$pwr" = "$R787_WANT_PWR" ] && [ "$gpc" = "$R787_WANT_GPC" ] && [ "$mem" = "$R787_WANT_MEM" ]; }

# Stop what is running of R787_QUIESCE and remember exactly that; idempotent. Under the chain the unit finds them already
# stopped, stops nothing and so restarts nothing: the chain restarts them once, after its daily restore.
# Call as a plain statement, NEVER inside $(...): R787_STOPPED must survive in the caller's shell. The message for the
# log is left in R787_QMSG.
r787_quiesce(){ local c
  for c in $R787_QUIESCE; do
    [ "$(sudo docker inspect -f '{{.State.Running}}' "$c" 2>/dev/null)" = true ] || continue
    sudo docker stop -t 30 "$c" >/dev/null 2>&1 && R787_STOPPED="$R787_STOPPED $c"; done
  R787_QMSG="direct :8022 clients stopped:${R787_STOPPED:- none (none running, or the chain already stopped them)}"; }
# Restart only what r787_quiesce stopped (same calling rule; message in R787_QMSG). Hermes' config is not touched (no
# hermes-set-model.sh: nothing is promoted).
r787_unquiesce(){ local was=$R787_STOPPED; R787_STOPPED=
  if [ -z "$was" ]; then R787_QMSG="no direct client to restart"
  elif sudo docker start $was >/dev/null 2>&1; then R787_QMSG="restarted:$was"
  else R787_QMSG="RESTART FAILED:$was (start by hand: sudo docker start$was)"; fi; }

# Report-only: request headers in a container log that carry no min_tokens, other than the first header (the launcher's
# own /completions warm-up). Every fn_bench request forces its length with min_tokens, so anything else is foreign.
# Regex = r736-std-workloads.sh's (any endpoint, stream or not).
r787_foreign(){ python3 - "$1" <<'PY'
import re, sys
t = open(sys.argv[1], errors="replace").read()
h = re.findall(r"INFO:\s+#(\d+) (?:chat/)?completions(?: \([\w-]+\))?: [\d,]+ prompt tokens ·\s+(.*?)(?=\n\S|\Z)", t, re.S)
if not h:
    print("?"); raise SystemExit
first = min(int(n) for n, _ in h)
print(sum(1 for n, rest in h if int(n) != first and "min_tokens" not in rest))
PY
}
