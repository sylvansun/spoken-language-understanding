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
    def load_dataset(cls, data_path, recall=4):
        dataset = json.load(open(data_path, "r"))
        # print(len(dataset))
        examples = []
        for di, data in enumerate(dataset):  # data is the sentences from one person
            ex = None
            for ui, utt in enumerate(data):
                ex = cls(utt, f"{di}-{ui}")
                examples.append(ex)
            refill = recall - (len(data) % 4)
            for i in range(refill):
                examples.append(ex)
        return examples

    def __init__(self, ex: dict, did):
        super(Example, self).__init__()
        self.ex = ex
        self.did = did
        self.utt = ex["asr_1best"]
        self.slot = {}
        for label in ex["semantic"]:
            # print("\n")
            # print("label:", label)
            act_slot = f"{label[0]}-{label[1]}"
            # print(act_slot)
            if len(label) == 3:  # Actually this is always true
                # print(label[2])
                self.slot[act_slot] = label[2]
        self.tags = ["O"] * len(self.utt)
        # print(self.tags)
        for slot in self.slot:
            value = self.slot[slot]
            # print(value)
            bidx = self.utt.find(value)
            if bidx != -1:
                self.tags[bidx : bidx + len(value)] = [f"I-{slot}"] * len(value)
                self.tags[bidx] = f"B-{slot}"
        self.slotvalue = [f"{slot}-{value}" for slot, value in self.slot.items()]
        self.input_idx = [Example.word_vocab[c] for c in self.utt]
        l = Example.label_vocab  # utils.vocab.LabelVocab object
        # print(l)
        self.tag_id = [l.convert_tag_to_idx(tag) for tag in self.tags]
