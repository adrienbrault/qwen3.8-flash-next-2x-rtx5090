#!/usr/bin/env bash
# Fails when README.md, THIRD_PARTY.md or docs/*.md contain mannered prose: evaluative or promotional words,
# rhetorical devices, exclamation marks, questions in running text. CLAUDE.md "Prose" lists the rule; this is the
# enforcement, run by check-public-hygiene.sh before every commit and by hand with `scripts/check-prose.sh [files]`.
# A line that must keep a flagged word (a quoted error string, a proper name) carries `prose-ok: <reason>`.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
FILES=("$@"); [ ${#FILES[@]} -eq 0 ] && FILES=(README.md THIRD_PARTY.md docs/*.md)
WORDS='blazing|robust|battle-tested|seamless|powerful|gold standard|huge|excited|exciting|honest|honestly|in anger|worthy|dumb|dumber|in this house|the job the box exists for|in the same class|the one that matters|that matters|read it as|provably|confident wrong|confidently wrong|coin-flip|the equal of|for good measure|needless to say|it turns out|as it happens|in short|simply put|truly|frankly|to be fair|of course|arguably|remarkably|surprisingly|impressive|impressively|elegant|beautiful|clever|magic|the good news|the bad news|the catch|the kicker|punchline|make no mistake|worth noting|notably|interestingly|unfortunately|fortunately|sadly|happily|no surprise|not a surprise|painful|cheap trick|sanity|insane|crazy|wild|nasty|ugly|pretty much|pretty good|pretty fast|a lot of|lots of|tons of|massive|massively|dramatic|dramatically|drastic|drastically|whopping|merely|the whole point|the point is|what this means is|in other words|put differently|wins|winner|loses|loser|beats|in its favour|in this seat.s favour|had it backwards|blamed the engine|in flight|in the hunt|the hunt|felt fine|looks exactly like|exactly like|no reason to|there is no reason|the answer is|answer:|question was whether|the question is|so the verdict|the verdict is|strengthens rather than weakens|not a tuning miss|structural, not|different worlds|a world|another world|world-class|state of the art|state-of-the-art|cutting-edge|next-level|game-changer|no-brainer|low-hanging|silver bullet|holy grail|rabbit hole|under the hood|at the end of the day|moving parts|sweet spot|knee|cliff|free lunch|the trick|tricks|hack|hacky|kludge|band-aid|bandaid|footgun|foot-gun'
RE_WORDS="(^|[^A-Za-z/_.-])(${WORDS})([^A-Za-z/_-]|$)"
hits=0
for f in "${FILES[@]}"; do
  [ -f "$f" ] || continue
  awk -v f="$f" -v re="$RE_WORDS" '
    BEGIN { fence = 0 }   # matching is done on tolower(line): BSD awk has no IGNORECASE
    /^```/ { fence = !fence; next }
    fence { next }
    /prose-ok:/ { next }
    /^\[[A-Za-z0-9_-]+\]: / { next }              # reference-style link definitions
    {
      line = $0
      lc = tolower(line)
      if (match(lc, re)) { print f ":" NR ": mannered word: " substr(lc, RSTART, RLENGTH) "  |  " line; bad = 1 }
      if (line ~ /(^|[^!=\[])![^=\[]/ || line ~ /![[:space:]]*$/) { print f ":" NR ": exclamation mark  |  " line; bad = 1 }
      if (line !~ /^\|/ && line !~ /^    / && line !~ /`[^`]*\?[^`]*`/ && line ~ /\?([[:space:]]|\*|$)/) { print f ":" NR ": question in running text  |  " line; bad = 1 }
    }
    END { exit bad ? 1 : 0 }
  ' "$f" || hits=1
done
if [ "$hits" = 1 ]; then echo "check-prose: FAIL (see lines above)"; exit 1; fi
echo "check-prose: OK"
