"""Mini-roundabouts: a mapped highway=mini_roundabout node (Overture's infrastructure layer) makes every
incoming arm give way to the circle (furniture.junction_controls), instead of the guessed signals and stops."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import furniture  # noqa: E402
import roads  # noqa: E402
import secrossings  # noqa: E402
from test_secrossings import Frame, Infra  # noqa: E402

HERE = np.array([[100.0, 100.0]])
FAR = np.array([[9000.0, 9000.0]])


def road(fid, cls, dx, dy, c=(100.0, 100.0), oneway=False, urban=True, width=10.0):
    pts = np.array([[c[0], c[1], 0.0], [c[0] + dx * 80.0, c[1] + dy * 80.0, 0.0]])
    return (fid, roads.Spec(width, "paved", urban, cls, oneway, dir=1 if oneway else 0), pts, 0.0)


def arm(fid, cls, dx, dy, oneway=False, urban=True, c=(100.0, 100.0)):
    pts = np.array([[c[0], c[1], 0.0], [c[0] + dx * 80.0, c[1] + dy * 80.0, 0.0]])
    return (fid, roads.Spec(10.0, "paved", urban, cls, oneway, dir=1 if oneway else 0),
            np.array([dx, dy], float), pts[0], True, 80.0)


def cross_roads():
    return [road(f"a{i}", "tertiary", dx, dy) for i, (dx, dy) in enumerate([(1, 0), (-1, 0), (0, 1), (0, -1)])]


def controls(arms_, **kw):
    return sorted((a[0], w) for a, w in furniture.junction_controls(arms_, None, **kw))


class ControlsTest(unittest.TestCase):
    def arms(self):
        return [arm(f"a{i}", "tertiary", dx, dy) for i, (dx, dy) in enumerate([(1, 0), (-1, 0), (0, 1), (0, -1)])]

    def test_every_incoming_arm_gives_way_to_the_circle(self):
        got = furniture.junction_controls(self.arms(), None, minis=HERE)
        self.assertEqual(sorted((a[0], w) for a, w in got), [(f"a{i}", furniture.G_YIELD) for i in range(4)])

    def test_it_beats_the_guessed_signals(self):
        self.assertEqual([w for _, w in controls(self.arms())], [furniture.G_SIGNAL] * 4,
                         "without the node the four-way is guessed signalled")
        self.assertEqual(controls(self.arms(), signals=np.zeros((0, 2)), minis=HERE),
                         [(f"a{i}", furniture.G_YIELD) for i in range(4)])

    def test_an_outbound_one_way_gets_nothing(self):
        arms = [arm("in", "tertiary", 1, 0), arm("out", "tertiary", -1, 0, oneway=True),
                arm("s", "residential", 0, -1)]
        self.assertTrue(furniture._outbound(arms[1]))
        self.assertEqual(controls(arms, minis=HERE), [("in", furniture.G_YIELD), ("s", furniture.G_YIELD)])

    def test_no_node_nearby_changes_nothing(self):
        without = controls(self.arms())
        self.assertEqual(controls(self.arms(), minis=FAR), without)
        self.assertEqual(controls(self.arms(), minis=None), without)

    def test_a_rural_junction_is_controlled_too(self):
        rs = [road(f"r{i}", "tertiary", dx, dy, urban=False) for i, (dx, dy) in
              enumerate([(1, 0), (-1, 0), (0, -1)])]
        self.assertEqual(furniture.junction_props(rs, (0, 0, 256, 256)), [], "elsewhere: no signs in the country")
        rows = furniture.junction_props(rs, (0, 0, 256, 256), minis=HERE)
        self.assertEqual(sorted(r[0] for r in rows), [furniture.G_YIELD] * 3)


class GiveWayArmsTest(unittest.TestCase):
    def roads(self):
        def seg(fid, a, b):
            return (fid, roads.Spec(10.0, "paved", True, "tertiary", False),
                    np.array([[a[0], a[1], 0.0], [b[0], b[1], 0.0]]), 0.0)
        return [seg("r1", (-80.0, 0.0), (0.0, 0.0)), seg("r2", (0.0, 0.0), (0.0, 80.0)),
                seg("r3", (0.0, 0.0), (400.0, 0.0)), seg("r4", (400.0, 0.0), (480.0, 0.0)),
                seg("r5", (400.0, 0.0), (400.0, 80.0))]

    def test_only_the_mini_roundabout_junctions_arms(self):
        got = furniture.give_way_arms(self.roads(), np.zeros((0, 2)), minis=np.array([[0.0, 0.0]]), only_minis=True)
        self.assertTrue(got, "the mini-roundabout junction's arms give way")
        self.assertEqual({g[0] for g in got}, {"r1", "r2", "r3"}, sorted({g[0] for g in got}))

    def test_without_a_mini_roundabout_nothing_is_painted(self):
        self.assertEqual(furniture.give_way_arms(self.roads(), np.zeros((0, 2)), minis=None, only_minis=True), [])


class NodesTest(unittest.TestCase):
    def test_reads_the_mini_roundabout_nodes(self):
        got = secrossings.nodes(Frame(), [Infra([
            ("mini_roundabout", (1.0, 2.0), {}),
            ("give_way", (3.0, 4.0), {"direction": "forward"}),
            ("crossing", (5.0, 6.0), {"crossing": "unmarked"}),
        ])])
        np.testing.assert_allclose(got["mini_roundabouts"], [[10.0, 20.0]])
        self.assertEqual(len(got["priority"]), 1)
        self.assertEqual(len(got["crossings"]), 1)

    def test_none_when_the_layer_cannot_be_read(self):
        self.assertIsNone(secrossings.nodes(Frame(), [None]))


class ReadTest(unittest.TestCase):
    """read() joins the Layercake file's mini-roundabout nodes (md.minis) to the infrastructure's."""
    @staticmethod
    def cell(b):
        return SimpleNamespace(bounds=lambda: b)

    @staticmethod
    def md(infra, minis):
        return SimpleNamespace(cell=lambda k, layers, kind: SimpleNamespace(layers={"infrastructure": infra}),
                               minis=minis if callable(minis) else (lambda bounds: minis))

    def test_the_layercake_minis_join_the_infrastructures(self):
        md = self.md(Infra([("crossing", (1.0, 2.0), {"crossing": "unmarked"})]), np.array([[0.5, 0.5]]))
        got = secrossings.read(md, Frame(), [self.cell((0.0, 0.0, 1.0, 1.0))])
        np.testing.assert_allclose(got["mini_roundabouts"], [[5.0, 5.0]])
        self.assertEqual(len(got["crossings"]), 1)

    def test_no_layercake_minis_keeps_the_infrastructures(self):
        md = self.md(Infra([("mini_roundabout", (1.0, 2.0), {})]), np.zeros((0, 2)))
        got = secrossings.read(md, Frame(), [self.cell((0.0, 0.0, 1.0, 1.0))])
        np.testing.assert_allclose(got["mini_roundabouts"], [[10.0, 20.0]])

    def test_minis_failing_costs_only_the_circles(self):
        def down(bounds):
            raise RuntimeError("down")
        md = self.md(Infra([("crossing", (1.0, 2.0), {})]), down)
        got = secrossings.read(md, Frame(), [self.cell((0.0, 0.0, 1.0, 1.0))])
        self.assertEqual(len(got["crossings"]), 1)
        self.assertEqual(len(got["mini_roundabouts"]), 0)

    def test_no_infrastructure_still_none(self):
        md = self.md(None, np.array([[0.5, 0.5]]))
        self.assertIsNone(secrossings.read(md, Frame(), [self.cell((0.0, 0.0, 1.0, 1.0))]))


class MiniIslandsTest(unittest.TestCase):
    """The raised island at a mapped mini-roundabout: centred on the node, sized by the narrowest arm, a kerb ring
    (sett) round a low cobble dome instead of paint."""
    def test_an_island_at_the_mapped_node(self):
        got = furniture.mini_islands(cross_roads(), HERE)
        self.assertEqual(len(got), 1)
        np.testing.assert_allclose(got[0]["node"][:2], [100.0, 100.0])
        self.assertAlmostEqual(got[0]["r"], 4.5)   # (width 10: 10 / 2 - 0.5)

    def test_the_narrowest_arms_width_sizes_it(self):
        roads_ = [road("a", "tertiary", 1, 0, width=6.0), road("b", "tertiary", -1, 0, width=12.0),
                  road("c", "tertiary", 0, 1, width=12.0)]
        self.assertAlmostEqual(furniture.mini_islands(roads_, HERE)[0]["r"], 2.5, places=1)   # (6 / 2 - 0.5)

    def test_no_junction_or_a_far_node_is_nothing(self):
        self.assertEqual(furniture.mini_islands([road("a", "tertiary", 1, 0)], HERE), [])
        self.assertEqual(furniture.mini_islands(cross_roads(), FAR), [])
        self.assertEqual(furniture.mini_islands(cross_roads(), None), [])

    def test_walks_kerb_ring_and_dome(self):
        walks = roads.mini_island_walks(furniture.mini_islands(cross_roads(), HERE), (-1e3, -1e3, 1e3, 1e3))
        discs = [w for w in walks if "disc" in w]
        bands = [w for w in walks if "pts" in w]
        self.assertEqual(len(discs), 1)
        self.assertTrue(bands)
        ring = np.asarray(discs[0]["disc"]["ring"], np.float64)
        r = np.hypot(ring[:, 0] - 100.0, ring[:, 1] - 100.0)
        self.assertAlmostEqual(float(r.mean()), 4.5 - furniture.MINI_KERB_W, places=2)
        self.assertEqual(discs[0]["disc"]["mat"], 7)    # (sett, the cobble dome)
        self.assertAlmostEqual(discs[0]["disc"]["h"], furniture.MINI_ISLAND_H + furniture.MINI_KERB_W * 0.015)
        self.assertAlmostEqual(discs[0]["disc"]["dome"], min(furniture.MINI_DOME_MAX_M,
                                                             furniture.MINI_DOME_PER_M * 4.5))
        band = np.vstack([np.asarray(w["pts"], np.float64) for w in bands])
        rb = np.hypot(band[:, 0] - 100.0, band[:, 1] - 100.0)
        self.assertAlmostEqual(float(rb.mean()), 4.5 - furniture.MINI_KERB_W / 2, places=1)
        self.assertTrue(all(w["h"] == furniture.MINI_ISLAND_H and w["mat"] == 7 and w["fid"] == "mini"
                            for w in bands))


if __name__ == "__main__":
    unittest.main()
