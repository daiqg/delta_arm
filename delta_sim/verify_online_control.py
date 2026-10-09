"""Check online timing isolation, analytic limits and planar paddle geometry."""
import unittest
import xml.etree.ElementTree as ET
import numpy as np
import kinematics as kin
from online_control import (OnlineController, StrokePlan, safe_command, safe_plan,
                            flat_paddle_model, HOME_PLAT)
from run_hitting import HittingSim


class OnlineChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sim = HittingSim()

    def test_analytic_feedforward_matches_finite_difference(self):
        m = self.sim.model
        gain = -m.actuator_biasprm[0,2]/m.actuator_gainprm[0,0]
        rng = np.random.default_rng(17)
        for _ in range(40):
            p = HOME_PLAT+rng.uniform(-.007,.007,3)
            v = rng.uniform(-.08,.08,3)
            q = kin.ik_safe(p)
            expected = q+gain*(kin.jacobian(q,eps=1e-7)@v)
            np.testing.assert_allclose(safe_command(m,p,v),expected,atol=2e-7)

    def test_vector_limits_agree_with_scalar_and_reject_overspeed(self):
        m = self.sim.model
        for speed in (.3,1.,1.5):
            plan = StrokePlan(.1,(HOME_PLAT,np.zeros(3),np.zeros(3)),.6,
                              np.array([-.02,0.,-.217]),np.array([speed,0.,.2]))
            scalar = all(safe_command(m,*plan.at(t)[:2]) is not None
                         for t in np.arange(plan.start,plan.end+.002,.002))
            self.assertEqual(safe_plan(m,plan),scalar)
            if speed==1.5:
                self.assertFalse(scalar)

    def test_pending_plan_cannot_act_before_publication(self):
        c = OnlineController(self.sim.model)
        p = StrokePlan(.2,(HOME_PLAT,np.zeros(3),np.zeros(3)),.6,
                       np.array([-.02,0.,-.217]),np.array([1.,0.,.2]))
        c.pending=p; c.next_update=10.
        c.update(.19,self.sim.q_home)
        self.assertIsNone(c.plan)
        np.testing.assert_allclose(c.commanded_state(.19)[0],HOME_PLAT)
        c.update(.2,self.sim.q_home)
        self.assertIs(c.plan,p)

    def test_contact_cancels_unpublished_plan(self):
        c = OnlineController(self.sim.model)
        c.pending=object()
        c.update(.2,self.sim.q_home,contacted=True)
        self.assertIsNone(c.pending)
        self.assertIsNone(c.plan)

    def test_flat_face_dimensions(self):
        root = flat_paddle_model()
        mesh = root.find(".//mesh[@name='flat_blade_contact']")
        vertices=np.fromstring(mesh.get('vertex'),sep=' ').reshape(-1,3)
        np.testing.assert_allclose(np.ptp(vertices,axis=0),[.170,.150,.012])
        np.testing.assert_allclose(np.unique(vertices[:,2]),[-.006,.006])
        self.assertEqual(root.find(".//geom[@name='paddle_face']").get('mass'),'0.075')


if __name__ == '__main__':
    unittest.main()
