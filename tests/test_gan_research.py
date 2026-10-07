"""Checks for the new fit boundary, domain constraints and fair batch exposure."""
import csv
import tempfile
import unittest
from pathlib import Path
import numpy as np
from benchmark.common import dump, load, read_raw, sha
from gan_research.pilot import (FEATURES, HELD_SUBTYPES, balanced_fit_indices, batch_plan, prepare,
                                project, quotas, subtype_mask)


class ConditionalPilotTests(unittest.TestCase):
    def test_balanced_gan_fit_preserves_every_unique_real_source(self):
        labels = np.repeat([2,3,4],[12,3,1])
        selected = balanced_fit_indices(labels,11)
        self.assertEqual(set(selected),set(range(len(labels))))
        self.assertEqual(np.bincount(labels[selected],minlength=5).tolist(),[0,0,12,12,12])
        np.testing.assert_array_equal(selected,balanced_fit_indices(labels,11))

    def test_subtype_holdout_moves_entire_full_feature_group(self):
        subtypes = np.array(['back','neptune','ipsweep','normal'])
        fp = np.array(['shared','shared','probe','normal'])
        np.testing.assert_array_equal(subtype_mask(subtypes,fp),[True,True,False,False])

    def test_projection_enforces_domains_and_training_only_constants(self):
        real = np.tile(np.array([.2]*23),(2,1)); real[1,:] = .7
        real[:,1] = 0  # Empirical land constant.
        real[:,3] = [0,1]
        real[:,0] = [1,8]
        candidate = np.tile(np.array([1.234]*23),(2,1))
        candidate[:,0] = [-2.2,3.6]; candidate[:,3] = [.2,.9]
        candidate[:,6] = [-.3,.437]
        result = project(candidate,real)
        np.testing.assert_array_equal(result[:,0],[0,4])
        np.testing.assert_array_equal(result[:,1],[0,0])
        np.testing.assert_array_equal(result[:,3],[0,1])
        np.testing.assert_array_equal(result[:,6],[0,.44])
        candidate[0,4] = np.nan
        with self.assertRaisesRegex(ValueError,'Invalid synthetic'):
            project(candidate,real)

    def test_class_exposure_is_paired_across_different_pool_sizes(self):
        y1 = np.repeat(np.arange(5),[20,15,5,3,1])
        y2 = np.repeat(np.arange(5),[20,15,15,9,3])
        x1 = np.column_stack([y1,np.arange(len(y1))])
        x2 = np.column_stack([y2,np.arange(len(y2))])
        a,ya = batch_plan(x1,y1,11,3); b,yb = batch_plan(x2,y2,11,3)
        np.testing.assert_array_equal(ya,yb)
        np.testing.assert_array_equal(a[:,0],ya); np.testing.assert_array_equal(b[:,0],yb)
        self.assertLessEqual(np.ptp(np.bincount(ya,minlength=5)),1)
        with self.assertRaisesRegex(ValueError,'support'):
            batch_plan(x1[y1 != 4],y1[y1 != 4],11,3)

    def test_doses_use_each_clients_real_support(self):
        y = np.tile(np.arange(5),7); clients = np.r_[np.zeros(15,dtype=int),np.ones(20,dtype=int)]
        self.assertEqual(quotas(y,clients,.25),[[0,0,1,1,1],[0,0,1,1,1]])
        self.assertEqual(quotas(y,clients,1),[[0,0,3,3,3],[0,0,4,4,4]])

    def test_prepare_refits_without_withheld_subtypes_or_old_validation(self):
        attacks = ['normal','neptune','back','ipsweep','portsweep','guess_passwd','warezclient','rootkit','loadmodule']
        records = []
        for label,attack in enumerate(attacks):
            for j in range(10):
                row = ['0']*43; row[1:4] = ['tcp','http','SF']; row[41] = attack
                row[0] = str((1000 if attack in HELD_SUBTYPES else 0)+label*20+j)
                records.append(row)
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); raw = root/'train.txt'
            with raw.open('w',newline='') as f: csv.writer(f).writerows(records)
            parsed = read_raw(raw); va = np.arange(0,len(records),10)
            tr = np.setdiff1d(np.arange(len(records)),va)
            previous,prior_local,evidence,local = [root/name for name in ['previous','prior-local','evidence','local']]
            previous.mkdir(); prior_local.mkdir()
            np.savez_compressed(prior_local/'development.npz',train_ids=parsed['ids'][tr],val_ids=parsed['ids'][va],
                                y_train=parsed['y'][tr],y_val=parsed['y'][va],clients=np.arange(len(tr))%2)
            dump(previous/'protocol.yaml',dict(development_sha256=sha(prior_local/'development.npz')))
            prepare(raw,previous,prior_local,evidence,local)
            prep = load(evidence/'preprocessing.json')
            with np.load(local/'development.npz') as d:
                self.assertEqual(set(prep['fit_ids']),set(d['core_ids']))
                self.assertFalse(set(d['core_ids'])&set(d['hard_ids']))
                self.assertFalse(set(d['core_ids'])&set(d['mixed_ids']))
                self.assertLess(prep['data_max'][0],1000)
                self.assertTrue(np.any(d['x_hard'][:,0] > 1))
                self.assertEqual(set(d['y_hard']),set(range(5)))
            with self.assertRaisesRegex(ValueError,'overwritten'):
                prepare(raw,previous,prior_local,evidence,local)


if __name__ == '__main__': unittest.main()
