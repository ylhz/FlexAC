from .image_base import ImageBaseDataset
from ..smp import *

from .chair_change import CHAIR, print_metrics, save_hallucinated_words

class COCO_Caption_Scorer():
    def __init__(self, ref, gt):
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.rouge.rouge import Rouge
        from pycocoevalcap.cider.cider import Cider

        self.ref = ref
        self.gt = gt
        print('setting up scorers...')
        self.scorers = [
            (Bleu(4), ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4']),
            (Rouge(), 'ROUGE_L'),
            (Cider(), 'CIDEr'),
        ]

    def compute_scores(self):
        total_scores = {}
        for scorer, method in self.scorers:
            print('computing %s score...' % (scorer.method()))
            score, scores = scorer.compute_score(self.gt, self.ref)
            if isinstance(method, list):
                for sc, scs, m in zip(score, scores, method):
                    print('%s: %0.3f' % (m, sc * 100))
                total_scores['Bleu'] = [x * 100 for x in score]
            else:
                print('%s: %0.3f' % (method, score * 100))
                total_scores[method] = score * 100

        print('*****DONE*****')
        for key, value in total_scores.items():
            print('{}:{}'.format(key, value))
        return total_scores


class ImageCaptionDataset(ImageBaseDataset):

    TYPE = 'Caption'

    DATASET_URL = {
        'COCO_VAL': 'https://opencompass.openxlab.space/utils/VLMEval/COCO_VAL.tsv',
    }

    DATASET_MD5 = {
        'COCO_VAL': '72a5079dead060269ac222c5aa5128af',
    }

    def load_data(self, dataset):
        data = super().load_data(dataset)
        if 'question' not in data:
            data['question'] = [(
                'Please describe this image in general. Directly provide the description, '
                'do not include prefix like "This image depicts". '
            )] * len(data)
        return data

    # It returns a dictionary of scores
    @classmethod
    def evaluate(self, eval_file, **kwargs):
        data = load(eval_file)
        lt = len(data)
        lines = [data.iloc[i] for i in range(lt)]
        ref, gt = {}, {}
        for i, line in enumerate(lines):
            ref[str(i)] = [str(line['prediction'])]
            gt[str(i)] = eval(line['answer'])

        scorer = COCO_Caption_Scorer(ref, gt)
        coco_caption_score_dict = scorer.compute_scores()
        score_pth = eval_file.replace('.xlsx', '_score.json')
        dump(coco_caption_score_dict, score_pth)
        return coco_caption_score_dict

class FlexACPairs(ImageBaseDataset):

    TYPE = 'Caption'

    DATASET_URL = {
        'COCO_VAL': 'https://opencompass.openxlab.space/utils/VLMEval/COCO_VAL.tsv',
    }

    DATASET_MD5 = {
        'COCO_VAL': '72a5079dead060269ac222c5aa5128af',
    }

    def load_data(self, dataset):
        data = super().load_data(dataset)
        prompt = ""
        if 'question' not in data:
            data['question'] = [(
                'Please describe this image in general. Directly provide the description, '
                'do not include prefix like "This image depicts". '
            )] * len(data)
        return data

    # It returns a dictionary of scores
    @classmethod
    def evaluate(self, eval_file, **kwargs):
        data = load(eval_file)
        lt = len(data)
        lines = [data.iloc[i] for i in range(lt)]
        ref, gt = {}, {}
        for i, line in enumerate(lines):
            ref[str(i)] = [str(line['prediction'])]
            gt[str(i)] = eval(line['answer'])

        scorer = COCO_Caption_Scorer(ref, gt)
        coco_caption_score_dict = scorer.compute_scores()
        score_pth = eval_file.replace('.xlsx', '_score.json')
        dump(coco_caption_score_dict, score_pth)
        return coco_caption_score_dict


class ChairDataset(ImageBaseDataset):
    TYPE = "Caption"
    DATASET_URL = {
        'ChairDataset': "",
    }

    DATASET_MD5 = {
        'ChairDataset': None,
    }

    def load_data(self, dataset):
        data = super().load_data(dataset)
        # if 'question' not in data:
        #     data['question'] = [(
        #         'Please describe this image in general. Directly provide the description, '
        #         'do not include prefix like "This image depicts". '
        #     )] * len(data)
        return data

    def dump_image(self, line):
        os.makedirs(self.img_root, exist_ok=True)

        if 'image' in line:
            if isinstance(line['image'], list):
                tgt_path = []
                assert 'image_path' in line
                for img, im_name in zip(line['image'], line['image_path']):
                    path = osp.join(self.img_root, im_name)
                    if not read_ok(path):
                        decode_base64_to_image_file(img, path)
                    tgt_path.append(path)
            else:
                tgt_path = osp.join(self.img_root, f"{line['index']}.png")
                if not read_ok(tgt_path):
                    decode_base64_to_image_file(line['image'], tgt_path)
                tgt_path = [tgt_path]
        else:
            assert 'image_path' in line
            tgt_path = toliststr(line['image_path'])

        return tgt_path
    
    @classmethod
    def evaluate(self, eval_file, **kwargs):
        # data = load(eval_file)
        # lt = len(data)
        # lines = [data.iloc[i] for i in range(lt)]
        # ref, gt = {}, {}
        # for i, line in enumerate(lines):
        #     ref[str(i)] = [str(line['prediction'])]
        #     gt[str(i)] = eval(line['answer'])

        cache = os.environ.get('Chair_cache')
        coco_path = os.environ.get('Chair_coco_path')
        image_id_key = 'image_path'
        caption_key = 'prediction'

        # # 将xlsx文件转换为jsonl文件
        # save_path = eval_file.replace('.xlsx', '.jsonl')
        # data = load(eval_file)
        # data.to_json(save_path, orient='records', lines=True)
        

        if cache and os.path.exists(cache):
            evaluator = pickle.load(open(cache, 'rb'))
            # print(f"loaded evaluator from cache: {args.cache}")
        else:
            print(f"cache not setted or not exist yet, building from scratch...")
            evaluator = CHAIR(coco_path)
            pickle.dump(evaluator, open(cache, 'wb'))
            print(f"cached evaluator to: {cache}")
        
        cap_dict, all_data = evaluator.compute_chair(eval_file, image_id_key, caption_key)


        # # 绘制分布直方图
        # plot_save_path = save_path.replace('.jsonl', '_histogram.png')
        # plot_histogram(all_data, bins=100, save_path=plot_save_path)

        
        score_pth = eval_file.replace('.xlsx', '_eval.json')
        dump(cap_dict, score_pth)

        score_pth = eval_file.replace('.xlsx', '_score.csv')
        # save

        keys, res = print_metrics(cap_dict)
        # save key,res to csv, 第一行为keys，第二行为res
        with open(score_pth, 'w') as f:
            f.write(','.join(keys) + '\n')
            f.write(','.join([str(x) for x in res]) + '\n')


        return cap_dict['overall_metrics']