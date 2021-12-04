import string
import editdistance
import numpy as np

class Evaluate:

    def ocr_metrics(self, predicts, ground_truth):
        """Calculate Character Error Rate (CER), Word Error Rate (WER) and Sequence Error Rate (SER)"""

        if len(predicts) == 0 or len(ground_truth) == 0:
            return (1, 1, 1)

        cer, wer, ser = [], [], []

        for (pd, gt) in zip(predicts, ground_truth):
            pd, gt = pd.lower(), gt.lower()

            pd_cer, gt_cer = list(pd), list(gt)
            dist = editdistance.eval(pd_cer, gt_cer)
            cer.append(dist / (max(len(pd_cer), len(gt_cer))))

            pd_wer, gt_wer = pd.split(), gt.split()
            dist = editdistance.eval(pd_wer, gt_wer)
            wer.append(dist / (max(len(pd_wer), len(gt_wer))))

            pd_ser, gt_ser = [pd], [gt]
            dist = editdistance.eval(pd_ser, gt_ser)
            ser.append(dist / (max(len(pd_ser), len(gt_ser))))

        metrics = [cer, wer, ser]
        metrics = np.mean(metrics, axis=1)

        return metrics
