"""Markdown QA for docs/: verify GFM table column counts, relative links, and anchors."""
import glob
import os
import re


def unescaped_pipes(line: str) -> int:
    """Count '|' characters that are NOT escaped as '\\|'."""
    n = 0
    for i, ch in enumerate(line):
        if ch != "|":
            continue
        # count only if the preceding char is not a backslash
        if i == 0 or line[i - 1] != "\\":
            n += 1
    return n


def check_tables() -> int:
    bad = 0
    for f in sorted(glob.glob("docs/*.md")):
        lines = open(f, encoding="utf-8").read().split("\n")
        in_tbl = False
        hdr = 0
        for i, line in enumerate(lines, 1):
            if line.strip().startswith("|"):
                n = unescaped_pipes(line)
                if not in_tbl:
                    in_tbl = True
                    hdr = n
                elif n != hdr:
                    print("  MISMATCH %s:%d (%d vs %d): %s" % (f, i, n, hdr, line[:60]))
                    bad += 1
            else:
                in_tbl = False
    return bad


def check_links():
    for f in ["docs/README.md", "docs/PLAN_REVIEW.md", "docs/BASELINE_SNAPSHOT.md"]:
        base = os.path.dirname(f)
        text = open(f, encoding="utf-8").read()
        for m in re.finditer(r"\]\((\./[^)#]+)", text):
            target = os.path.normpath(os.path.join(base, m.group(1)))
            print("  %s  %s" % ("OK  " if os.path.exists(target) else "MISS", m.group(1)))


def check_anchors():
    src = open("docs/PLAN_REVIEW.md", encoding="utf-8").read()
    heads = set()
    for h in re.findall(r"^##+\s+(.*)$", src, re.M):
        slug = re.sub(r"[^a-z0-9 -]", "", h.lower()).replace(" ", "-")
        heads.add(slug)
    rd = open("docs/README.md", encoding="utf-8").read()
    for a in sorted(set(re.findall(r"PLAN_REVIEW\.md#([a-z0-9-]+)", rd))):
        print("  %s  #%s" % ("OK  " if a in heads else "MISS", a))


if __name__ == "__main__":
    print("== tables ==")
    n = check_tables()
    print("  column mismatches:", n)
    print("== relative links ==")
    check_links()
    print("== anchors ==")
    check_anchors()