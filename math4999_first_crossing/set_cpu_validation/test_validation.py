import unittest
from fractions import Fraction as F
from . import models, oracles, campaign

class ValidationTests(unittest.TestCase):
    def test_strict_equal_and_censor(self):
        self.assertEqual(models.interval_survival([0]*3, [(F(1,5),F(1,5))]*3, F(1,5))[0], {4})
        self.assertEqual(oracles.interval_oracle([0]*3, [(F(1,5),F(1,5))]*3, F(1,5))[0], {4})
    def test_fme_strict_and_one_sided(self):
        self.assertFalse(oracles.fme([([1],0,False),([-1],0,True)]))
        self.assertTrue(oracles.fme([([1],0,True)]))
        self.assertFalse(oracles.fme([([0],0,True)]))
        self.assertFalse(oracles.fme([([1],0,False),([2],0,True),([-1],0,False)]))
    def test_rectangle_and_radial(self):
        self.assertEqual(models.rectangle_min((F(1,10),F(1,5)),(0,F(1,5)),F(3,5))[0], F(16,2500))
        self.assertEqual(models.radial_survival(1,1,F(1,10),F(1,10),F(1,5),4)[0], {3,4,5})
    def test_vector_and_roster(self):
        a=[((1,0),(0,1))]*3; d=[((0,F(-1,10)),(0,F(1,10)))]*3
        self.assertEqual(models.vector_survival(a,d,F(1,5))[0], {3,4})
        self.assertEqual(oracles.vector_oracle(a,d,F(1,5))[0], {3,4})
        self.assertEqual(len(campaign.build_cases()),87)
    def test_invalid(self):
        with self.assertRaises(ValueError): models.interval_survival([1],[(1,0)],F(1,5))
        with self.assertRaises(ValueError): models.radial_survival(1,0,0,1,F(1,5),1)

class FailureAndCertificateTests(unittest.TestCase):
    def test_fme_resource_caps_and_deadline(self):
        from time import monotonic
        with self.assertRaises(oracles.LimitError): oracles.fme([([1,0],1,False),([0,1],1,False)],cap=1)
        with self.assertRaises(oracles.LimitError): oracles.fme([([1],1,False)],deadline=monotonic()-1)
    def test_fme_scaled_stronger_bound(self):
        self.assertFalse(oracles.fme([([1],2,False),([2],2,False),([-1],-1,True)]))
        self.assertTrue(oracles.fme([([1],2,False),([2],2,False),([-1],-1,False)]))
    def test_empty_survivors_and_input_lengths(self):
        self.assertEqual(models.interval_survival([0,0],[(1,1),(0,0)],F(1,5))[0],{1})
        with self.assertRaises(ValueError): models.interval_survival([1,1],[(0,0)],F(1,5))
        with self.assertRaises(ValueError): models.radial_survival(1,1,0,0,F(1,5),0)
    def test_rectangle_certificate_degenerate(self):
        rect=((F(3,10),F(3,10)),(0,0)); q,p=models.rectangle_min(*rect,F(3,5))
        self.assertEqual((q,p),(F(9,100),(F(3,10),F(0))))
        cert=campaign.rectangle_certificate(rect,F(3,5),q,p)
        self.assertTrue(all(v for v in cert.values() if isinstance(v,bool)))
        self.assertFalse(campaign.rectangle_certificate(rect,F(3,5),F(0),p)['objective'])
    def test_technical_stop_keeps_partial_and_existing_output(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        import json
        protocol=Path('evidence/math4999_next_step_20261105_001/CPU_PROTOCOL_V1.md')
        with tempfile.TemporaryDirectory(dir=".") as tmp:
            output=Path(tmp)/'new'
            cases=[{'id':'T01','kind':'T'}]
            with patch.object(campaign,'build_cases',return_value=cases), patch.object(campaign,'validate_case',side_effect=oracles.LimitError('test limit')):
                self.assertFalse(campaign.run(protocol,output,1))
            summary=json.loads((output/'summary.json').read_text(encoding='utf-8'))
            self.assertEqual((summary['status'],summary['technical_failures'],summary['planned']),('STOPPED_TECHNICAL_FAILURE',1,87))
            self.assertIn('test limit',(output/'cases.jsonl').read_text(encoding='utf-8'))
            self.assertTrue((output/'manifest.json').exists() and (output/'hashes.json').exists())
            with self.assertRaises(FileExistsError): campaign.run(protocol,output,1)
    def test_protocol_mismatch_creates_no_directory(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory(dir=".") as tmp:
            p=Path(tmp)/'bad'; p.write_text('bad'); out=Path(tmp)/'out'
            with self.assertRaises(ValueError): campaign.run(p,out,1)
            self.assertFalse(out.exists())


class ArchitectFMETests(unittest.TestCase):
    def test_strict_boundary_multivariate(self):
        rows=[([2,1],1,False),([-1,1],0,True),([0,1],F(1,3),False),([0,-1],F(-1,3),False)]
        self.assertFalse(oracles.fme(rows))
        rows[1]=([-1,1],0,False)
        self.assertTrue(oracles.fme(rows))
    def test_zero_row_and_one_sided(self):
        self.assertTrue(oracles.fme([([-2],-2,True),([0],0,False)]))
        self.assertFalse(oracles.fme([([-2],-2,True),([0],0,True)]))

class CaseAndLifecycleTests(unittest.TestCase):
    def test_frozen_roster_counts_and_ids(self):
        from collections import Counter
        cases=campaign.build_cases()
        self.assertEqual(Counter(c['kind'] for c in cases),{'A':48,'B':9,'G':18,'R':6,'P':6})
        self.assertEqual(len({c['id'] for c in cases}),87)
    def test_hole_and_fixed_theta_negative_control(self):
        cases={c['id']:c for c in campaign.build_cases()}
        hole=campaign.validate_case(cases['P04'],None)
        fixed=campaign.validate_case(cases['P05'],None)
        self.assertTrue(hole['passed']); self.assertEqual(hole['metrics']['times'],[1,3]); self.assertEqual(hole['metrics']['holes'],[2])
        self.assertTrue(fixed['passed']); self.assertEqual(fixed['metrics']['times'],[3]); self.assertEqual(fixed['details']['stepwise_reselection_negative_control'],{2,3})
    def test_failure_result_stops_and_retains_case(self):
        import tempfile, json
        from pathlib import Path
        from unittest.mock import patch
        with tempfile.TemporaryDirectory(dir='.') as tmp:
            output=Path(tmp)/'out'; protocol=Path('evidence/math4999_next_step_20261105_001/CPU_PROTOCOL_V1.md')
            with patch.object(campaign,'build_cases',return_value=[{'id':'T01','kind':'T'}]), patch.object(campaign,'validate_case',return_value={'id':'T01','kind':'T','passed':False}):
                self.assertFalse(campaign.run(protocol,output,1))
            summary=json.loads((output/'summary.json').read_text(encoding='utf-8'))
            self.assertEqual(summary['status'],'STOPPED_FAILURE'); self.assertEqual(summary['failed'],1)
    def test_reject_inexact_oracle_input(self):
        with self.assertRaises(ValueError): oracles.fme([([0.1],1,False)])
        with self.assertRaises(ValueError): oracles.fme([([1],1,1)])

class ReviewRegressionTests(unittest.TestCase):
    def test_sharp_old_g_uses_lower_only_piecewise_minimum(self):
        self.assertEqual(models.old_g_squared((F(3,10),F(3,10)),(0,0),F(4,5)),F(81,2500))
        self.assertEqual(models.old_g_squared((F(1,10),F(1,10)),(F(1,10),F(1,10)),F(1,2)),F(1,100))
        self.assertEqual(models.old_g_squared((1,1),(1,1),1),0)
    def test_censoring_removes_finite_guaranteed_right_endpoint(self):
        self.assertIsNone(campaign.metrics({3,4,5},4,{3,4,5})['finite_right_endpoint'])
        self.assertEqual(campaign.metrics({2},4,{2})['finite_right_endpoint'],2)
        self.assertIsNone(campaign.metrics({5},4,{5})['finite_right_endpoint'])
    def test_positive_threshold_is_required(self):
        with self.assertRaises(ValueError): models.interval_survival([1],[(0,0)],0)
        with self.assertRaises(ValueError): models.vector_survival([((1,0),(0,1))],[((0,0),)],0)
        with self.assertRaises(ValueError): models.radial_survival(1,1,0,0,0,1)
        with self.assertRaises(ValueError): oracles.interval_oracle([1],[(0,0)],0)
        with self.assertRaises(ValueError): oracles.vector_oracle([((1,0),(0,1))],[((0,0),)],0)

class FinalReviewTests(unittest.TestCase):
    def test_invalid_rectangles(self):
        bad=[((2,1),(0,1),F(1,2)),((0,1),(2,1),F(1,2)),((0,1),(0,1),-1),((0,1),(0,1),2),((-1,1),(0,1),0),((0,1),(-1,1),0)]
        for fn in (models.rectangle_min,models.old_g_squared):
            for x,y,k in bad:
                with self.subTest(fn=fn.__name__,x=x,y=y,k=k):
                    with self.assertRaises(ValueError): fn(x,y,k)
    def test_sharp_old_g_matches_frozen_reference_on_roster(self):
        from math4999_first_crossing.first_crossing import sharp_step_squared
        cases=[c for c in campaign.build_cases() if c['kind']=='G']
        self.assertEqual(len(cases),18)
        for c in cases:
            x,y=c['rectangle']
            self.assertEqual(models.old_g_squared(x,y,c['k']),sharp_step_squared(x[0],y[0],c['k']))
    def test_nontrivial_interval_oracle(self):
        self.assertEqual(oracles.interval_oracle([F(3,2)]*4,[(F(-1,10),F(1,10))]*4,F(1,5))[0],{2,3,4,5})
    def test_validation_detects_missing_event_from_implementations(self):
        from unittest.mock import patch
        cases=campaign.build_cases()
        a=next(c for c in cases if c['kind']=='A' and c['gain']==F(3,2) and c['interval']==(F(-1,10),F(1,10)) and c['H']==4)
        b=next(c for c in cases if c['kind']=='B' and c['matrix']==((1,0),(0,1)) and c['H']==3)
        for case,name,missing in ((a,'interval_survival',2),(b,'vector_survival',3)):
            original=getattr(models,name)
            def broken(*args, original=original, missing=missing):
                times,trace=original(*args); return times-{missing},trace
            with patch.object(models,name,side_effect=broken):
                self.assertFalse(campaign.validate_case(case,None)['passed'])
    def test_p_execution_reads_roster_parameters(self):
        case=next(c for c in campaign.build_cases() if c['id']=='P01')
        case['parameters']['intervals']=[(0,0)]*3
        self.assertFalse(campaign.validate_case(case,None)['passed'])
    def test_serialization_failure_has_no_phantom_pass(self):
        import tempfile,json
        from pathlib import Path
        from unittest.mock import patch
        with tempfile.TemporaryDirectory(dir='.') as tmp:
            output=Path(tmp)/'out'; protocol=Path('evidence/math4999_next_step_20261105_001/CPU_PROTOCOL_V1.md')
            original=campaign.serial
            def fail_result(obj):
                if isinstance(obj,dict) and obj.get('id')=='T01' and 'passed' in obj: raise TypeError('test serialization failure')
                return original(obj)
            with patch.object(campaign,'build_cases',return_value=[{'id':'T01','kind':'T'}]), patch.object(campaign,'validate_case',return_value={'id':'T01','kind':'T','passed':True}), patch.object(campaign,'serial',side_effect=fail_result):
                self.assertFalse(campaign.run(protocol,output,1))
            summary=json.loads((output/'summary.json').read_text(encoding='utf-8'))
            self.assertEqual((summary.get('status'),summary.get('passed'),summary.get('errors'),summary.get('attempted'),summary.get('written'),summary.get('not_run')),('STOPPED_ERROR',0,1,1,0,86))

    def test_validation_exception_is_error_not_limit_failure(self):
        import tempfile,json
        from pathlib import Path
        from unittest.mock import patch
        with tempfile.TemporaryDirectory(dir='.') as tmp:
            output=Path(tmp)/'out'; protocol=Path('evidence/math4999_next_step_20261105_001/CPU_PROTOCOL_V1.md')
            with patch.object(campaign,'build_cases',return_value=[{'id':'T01','kind':'T'}]), patch.object(campaign,'validate_case',side_effect=RuntimeError('unexpected')):
                self.assertFalse(campaign.run(protocol,output,1))
            summary=json.loads((output/'summary.json').read_text(encoding='utf-8'))
            self.assertEqual((summary['status'],summary['errors'],summary['technical_failures'],summary['attempted'],summary['written'],summary['not_run']),('STOPPED_ERROR',1,0,1,1,86))
