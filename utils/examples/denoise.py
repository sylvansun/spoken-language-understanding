import json

from utils.vocab import Vocab, LabelVocab
from utils.word2vec import Word2vecUtils
from utils.evaluator import Evaluator


class Example:
    @classmethod
    def configuration(cls, root, train_path=None, word2vec_path=None):
        cls.evaluator = Evaluator()
        cls.word_vocab = Vocab(padding=True, unk=True, filepath=train_path)
        cls.word2vec = Word2vecUtils(word2vec_path)
        cls.label_vocab = LabelVocab(root)

    @classmethod
    def load_dataset(cls, data_path):
        datas = json.load(open(data_path, "r"))
        examples = []
        for data in datas:
            for utt in data:
                ex = cls(utt)
                # assert False
                examples.append(ex)
        return examples

    def __init__(self, ex: dict):
        super(Example, self).__init__()
        self.ex = ex
        self.utt = ex["asr_1best"]
        self.denoise_utt = (
            ex["manual_transcript"]
            .replace("(unknown)", "")
            .replace("(side)", "")
            .replace("(dialect)", "")
            .replace("(robot)", "")
            .replace("noise", "")
        )

        # print(self.utt)
        self.slot = {}
        for label in ex["semantic"]:
            act_slot = f"{label[0]}-{label[1]}"
            # print(act_slot)
            if len(label) == 3:
                self.slot[act_slot] = label[2]
                # print(label[2])
        self.tags = ["O"] * len(self.denoise_utt)
        # print(self.tags)
        for slot in self.slot:
            value = self.slot[slot]
            bidx = self.denoise_utt.find(value)
            if bidx != -1:
                self.tags[bidx : bidx + len(value)] = [f"I-{slot}"] * len(value)
                self.tags[bidx] = f"B-{slot}"
        # print(self.tags)
        self.slotvalue = [f"{slot}-{value}" for slot, value in self.slot.items()]
        self.input_idx = [Example.word_vocab[c] for c in self.utt]
        self.denoise_idx = [Example.word_vocab[c] for c in self.denoise_utt]
        l = Example.label_vocab
        self.tag_id = [l.convert_tag_to_idx(tag) for tag in self.tags]
        # print(self.tag_id)
