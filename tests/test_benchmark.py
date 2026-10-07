"""Leakage boundary and evaluation semantics, independent of training speed."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from benchmark.common import assert_boundaries, group_split, client_assignments, metrics, read_raw


class BenchmarkBoundaryTests(unittest.TestCase):
    def test_holdout_cannot_enter_scaler_fit(self):
        with self.assertRaisesRegex(ValueError,'fit IDs'):
            assert_boundaries(['t1','t2'],['v'],['s'],['t1','v'],np.array([0,1]))
        with self.assertRaisesRegex(ValueError,'overlap'):
            assert_boundaries(['t1'],['t1'],['s'],['t1'],np.array([0]))

    def test_duplicate_groups_stay_together_and_cover_clients(self):
        y=np.repeat(np.arange(5),20); fp=np.array([f'{label}-{i//2}' for label in range(5) for i in range(20)])
        train,val=group_split(y,fp)
        self.assertFalse(set(fp[train])&set(fp[val]))
        assignment=client_assignments(y[train],fp[train])
        self.assertEqual(set(assignment),{0,1})
        audit=assert_boundaries(np.array([str(i) for i in train]),np.array([str(i) for i in val]),['external'],
                               np.array([str(i) for i in train]),assignment,fp[train],fp[val],['external-fp'])
        self.assertTrue(audit['client_partition_complete'])
        repeat=client_assignments(y[train],fp[train]); np.testing.assert_array_equal(assignment,repeat)
        bad=assignment.copy(); key=fp[train][0]; members=np.flatnonzero(fp[train]==key); bad[members]=[0,1]
        with self.assertRaisesRegex(ValueError,'spans clients'):
            assert_boundaries(np.array([str(i) for i in train]),np.array([str(i) for i in val]),['external'],
                              np.array([str(i) for i in train]),bad,fp[train],fp[val],['external-fp'])

    def test_conflicting_real_labels_are_rejected(self):
        y=np.repeat(np.arange(5),4); fp=np.array([str(i) for i in range(20)])
        fp[4]=fp[0]
        with self.assertRaisesRegex(ValueError,'Conflicting'):
            group_split(y,fp)

    def test_macro_scores_include_never_predicted_classes(self):
        y=np.arange(5); p=np.zeros((5,5)); p[:,0]=1
        m=metrics(y,p)
        self.assertAlmostEqual(m['accuracy'],.2)
        self.assertAlmostEqual(m['macro_f1'],(1/3)/5)
        self.assertEqual(m['never_predicted'],[1,2,3,4])
        self.assertEqual(m['binary_attack_recall'],0)
        self.assertEqual(m['normal_false_positive_rate'],0)

    def test_parser_rejects_unknown_attack_and_nonfinite_features(self):
        row=['0']*43; row[1:4]=['tcp','http','SF']; row[41]='unknown'
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'raw.txt'; p.write_text(','.join(row)+'\n')
            with self.assertRaisesRegex(ValueError,'Unmapped'): read_raw(p)
            row[41]='normal'; row[0]='nan'; p.write_text(','.join(row)+'\n')
            with self.assertRaisesRegex(ValueError,'Nonfinite'): read_raw(p)

    def test_raw_preparation_freezes_real_splits_and_training_scale(self):
        from benchmark.prepare import prepare
        from benchmark.common import load
        attacks=['normal','back','ipsweep','guess_passwd','rootkit']
        def records(test=False):
            rows=[]
            for label,attack in enumerate(attacks):
                for j in range(1 if test else 10):
                    row=['0']*43; row[1:4]=['tcp','http','SF']
                    row[0]=str(10000+label if test else label*100+j)
                    row[41]=attack; rows.append(','.join(row))
            return '\n'.join(rows)+'\n'
        with tempfile.TemporaryDirectory() as t:
            root=Path(t); tr=root/'train.txt'; te=root/'test.txt'
            tr.write_text(records()); te.write_text(records(True))
            e,l=root/'evidence',root/'local'; prepare(tr,te,e,l)
            with np.load(l/'development.npz') as d, np.load(l/'test.npz') as test:
                prep=load(e/'preprocessing.json')
                self.assertEqual(set(prep['fit_ids']),set(d['train_ids']))
                self.assertTrue(np.all(test['x'][:,0]>1))
                self.assertEqual(len(d['y_train'])+len(d['y_val']),50)
                self.assertTrue(set(d['train_ids']).isdisjoint(test['ids']))
            with self.assertRaisesRegex(ValueError,'overwritten'):
                prepare(tr,te,e,l)
