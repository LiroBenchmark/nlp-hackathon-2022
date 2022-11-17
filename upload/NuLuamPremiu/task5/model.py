# !pip install transformers
# !pip install sentencepiece
# !pip install datasets

from transformers import MT5ForConditionalGeneration, T5Tokenizer
from nltk.tokenize import wordpunct_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.tokenize import RegexpTokenizer
from tqdm.notebook import tqdm


class MyModel:
    def __init__(self):
        # do here any initializations you require

        self.DIAC_MAP = {
            "ț": "t",
            "ș": "s",
            "Ț": "T",
            "Ș": "S",
            "Ă": "A",
            "ă": "a",
            "Â": "A",
            "â": "a",
            "Î": "I",
            "î": "i",
        }

        self.GENERATE_DIAC_MAP = {
            "t": ["t", "ț"],
            "s": ["s", "ș"],
            "T": ["T", "Ț"],
            "S": ["S", "Ș"],
            "A": ["A", "Ă", "Â"],
            "a": ["a", "ă", "â"],
            "I": ["I", "Î"],
            "i": ["i", "î"],
        }

        self.fine_tune_diacrit_dict = {}  # Mandatory diacritics exchanges dictionary
        # Found words over the statistical threshhold

        self.known_mandatory_diacritics_set = []

    def remove_diacritics_prop(self, prop):
        for diac in self.DIAC_MAP:
            prop = prop.replace(diac, self.DIAC_MAP[diac])

        return prop

    # Banned list allows not to generate diacritics permutations on specific letters
    def generate_word_diacritics_perms(
        self, original_word, crt_length, generated_permutations, banned_list
    ):
        if crt_length == len(original_word):
            return generated_permutations

        if crt_length == 0:
            generated_permutations = [""]

        # define the brute letter
        original_letter = original_word[crt_length]

        new_generated_permutations = []

        # if we want to generte a permutation for a letter but we do not want to be in the banned list,
        # but for this we want to forck the tree undeterministically
        # otherwise insert a new letter to all current permutations
        if (
            original_letter in self.GENERATE_DIAC_MAP.keys()
            and original_letter not in banned_list
        ):
            diacritics_for_letter = self.GENERATE_DIAC_MAP[original_letter]

            for word in generated_permutations:
                for diacrit in diacritics_for_letter:
                    new_generated_permutations.append(word + diacrit)
        else:
            for word in generated_permutations:
                new_generated_permutations.append(word + original_letter)

        generated_permutations = new_generated_permutations

        return self.generate_word_diacritics_perms(
            original_word, crt_length + 1, generated_permutations, banned_list
        )

    # populates the fine-tune dictionary based on the sets of letters that goes more than the threshhold
    def populate_finetune_dict(self):
        for word in self.known_mandatory_diacritics_set:
            stripped_word = self.remove_diacritics_prop(word)

            self.fine_tune_diacrit_dict[stripped_word] = word

        return

    # Primeste outputul de la transformatorul vostru, si peste el face replaceurile corespunzatoare in-place
    # It receives the output from your transformer, and on top of it, it makes the appropriate in-place replacements
    def mandatory_diacritics(self, original_sentence, transformer_output):
        tokenizer = RegexpTokenizer(r"\w+")

        tokenized_transformer_output = tokenizer.tokenize(transformer_output)
        tokenized_original_sentence = tokenizer.tokenize(original_sentence)

        crt_token_idx = 0
        i = 0
        new_sentence = ""

        # searching the mandatory enter for the current token and returns it in its place
        while i < len(original_sentence):
            crt_char = original_sentence[i]

            if crt_char.isalnum():
                crt_original_token = tokenized_original_sentence[crt_token_idx]
                crt_transformer_token = tokenized_transformer_output[crt_token_idx]

                crt_token_idx += 1

                stripped_og_token = self.remove_diacritics_prop(crt_original_token)
                stripped_transf_token = self.remove_diacritics_prop(
                    crt_transformer_token
                )

                if stripped_og_token != stripped_transf_token:
                    final_token = stripped_og_token
                else:
                    final_token = crt_transformer_token

                i += len(final_token)

                if final_token in self.fine_tune_diacrit_dict:
                    new_sentence += self.fine_tune_diacrit_dict[final_token]
                else:
                    new_sentence += final_token
            else:
                new_sentence += crt_char

                i += 1

        return new_sentence

    def load(self, model_resource_folder):
        # we'll call this code before prediction
        # use this function to load any pretrained model and any other resource,
        # from the given folder path
        self.model = MT5ForConditionalGeneration.from_pretrained(
            "iliemihai/mt5-base-romanian-diacritics"
        )

        self.model.to("cuda")

        self.tokenizer = T5Tokenizer.from_pretrained(
            "iliemihai/mt5-base-romanian-diacritics"
        )

    # *** OPTIONAL ***
    def train(self, train_data_file, validation_data_file, model_resource_folder):
        cuv = dict()
        i = 0

        for el in validation_data_file:
            print(f"{i}")
            i += 1

            list_of_words = "".join(
                (char if char.isalpha() else " ") for char in el["text"]
            ).split()

            if len(list_of_words) < 1:
                continue

            list_of_words[0] = list_of_words[0].lower()
            for word in list_of_words:
                word_no_diac = word
                for diac in self.DIAC_MAP:
                    word_no_diac = word_no_diac.replace(diac, self.DIAC_MAP[diac])

                if word_no_diac in cuv:
                    if (
                        len([item for item in cuv[word_no_diac] if item[0] == word])
                        == 0
                    ):
                        cuv[word_no_diac].append((word, 1))
                    else:
                        el = [item for item in cuv[word_no_diac] if item[0] == word][0]
                        nr = el[1]
                        cuv[word_no_diac].remove(el)
                        nr = nr + 1
                        new_el = (word, nr)
                        cuv[word_no_diac].append(new_el)
                else:
                    cuv[word_no_diac] = [(word, 1)]

        for el in cuv:
            sum = 0
            for element in cuv[el]:
                sum = sum + element[1]

            for element in cuv[el]:
                if element[1] >= 0.95 * sum:
                    cuv[el] = [(element[0], 1)]
        resultList = list(cuv.items())

        final = set()

        # calculate the probabilities for every word and if one word appears with more
        # than 95% then we only keep that version of the word
        for el in cuv:
            if len(cuv[el]) == 1:
                word = cuv[el][0][0]
                final.add(word)
                final.add(word.capitalize())

        self.known_mandatory_diacritics_set = final
        self.populate_finetune_dict()

        # we'll call this function right after init
        # place here all your training code
        # at the end of training, place all required resources, trained model, etc in
        # the given model_resource_folder
        return

    def predict(self, input_file, output_file):
        # we'll call this function after the load()
        # use this place to run the prediction
        # the input is a file that does not contain diacritics
        # the output is a file that contains diacritics and,
        #    **is identical at character level** with the input file,
        #   excepting replaced diacritic letters
        final_output = ""
        i = 0

        with open(input_file, "r") as f:
            for line in f:
                print(f"{i}")
                i += 1

                inputs = self.tokenizer(
                    line, max_length=256, truncation=True, return_tensors="pt"
                )

                inputs.to("cuda")

                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                )
                output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                output = self.mandatory_diacritics(line, output)

                final_output += f"{output.strip()}\n"

        with open(output_file, "w") as f:
            f.write(final_output)
