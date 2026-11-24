from read_hand import apply_edit


def names(g):
    return [[e["tile"] for e in grp] for grp in g]


g = [[{"tile": "flower3", "conf": 1}, {"tile": "4dot", "conf": 1}], [{"tile": "2dot", "conf": 1}] * 2]
g[1] = [dict(e) for e in g[1]]
apply_edit(g, "2=4bam");  assert names(g) == [["flower3", "4bam"], ["2dot", "2dot"]]
apply_edit(g, "|2");      assert names(g) == [["flower3"], ["4bam"], ["2dot", "2dot"]]
apply_edit(g, "-1");      assert names(g) == [["4bam"], ["2dot", "2dot"]]
apply_edit(g, "+north");  assert names(g) == [["4bam"], ["2dot", "2dot", "north"]]
for bad in ("9=north", "1=dragon", "+nope"):
    try:
        apply_edit(g, bad); raise AssertionError(bad)
    except (ValueError, IndexError):
        pass
print("ok")
