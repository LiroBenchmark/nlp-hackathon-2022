import regex as re

class Evaluator():
    diacs = set(['ț', 'ș', 'Ț', 'Ș', 'Ă', 'ă', 'Â', 'â', 'Î', 'î'])
    non_diacs = set(['a', 't', 'i', 's'])
    target_chars = diacs.union(non_diacs)
    diac_map = {'ț': 't', 'ș': 's', 'Ț': 'T', 'Ș': 'S', 'Ă': 'A', 'ă': 'a', 'Â': 'A', 'â': 'a', 'Î': 'I', 'î': 'i'}

    def __init__(self, tbl_wordform_file=None):
        import os

        # load and process tbl.wordform.ro
        if tbl_wordform_file is None:
            tbl_wordform_file = "tbl.wordform.ro"
        if not os.path.exists(tbl_wordform_file):
            try: # try to download the file
                import requests
                url = "https://raw.githubusercontent.com/racai-ai/Rodna/master/data/resources/tbl.wordform.ro"
                r = requests.get(url, allow_redirects=True)
                open('tbl.wordform.ro', 'wb').write(r.content)
            except Exception as ex:
                raise Exception(f"'tbl.wordform.ro' file not found, I checked in [{os.path.abspath(tbl_wordform_file)}], please check that it exists, that you have an internet connection for auto-download, or manually specify an absolute path to this file! Exception raised: {str(ex)}")

        with open(tbl_wordform_file, "r", encoding="utf8") as f:
            lines = f.readlines()
        assert len(lines) > 0

        targets = {}
        for line in lines:
            if line.startswith("#") or line.strip() == "":
                continue
            line = line.replace("&acirc", "â").replace("&Acirc", "Â")
            line = line.replace("&tcedil;", "ț").replace("&Tcedil;", "Ț")
            line = line.replace("&scedil;", "ș").replace("&Scedil;", "Ș")
            line = line.replace("&abreve;", "ă").replace("&Abreve;", "Ă")
            line = line.replace("&icirc;", "î").replace("&Icirc;", "Î")
            if "&" in line:  # we're going to skip words with utf8 encoded chars that are not in our targets, there are only 233 of them anyway
                continue

            parts = line.split("\t")
            if parts[1] == "=":
                parts[1] = parts[0]

            surface = parts[0].lower()
            surface_nodiac = Evaluator.remove_diacritics(surface)
            root = parts[1]

            # check to see if it is a target word
            target_word = False
            for c in surface_nodiac:
                if c in Evaluator.non_diacs:
                    target_word = True
                    break
            if target_word:  # add only if there are possible diacritics to be added
                if surface_nodiac not in targets:
                    targets[surface_nodiac] = set()
                targets[surface_nodiac].add(surface)

        # filter target words
        self.target_words = set()
        self.strict_target_words = set()
        for word in targets:
            if len(targets[word]) >= 1:
                self.target_words.add(word)
            if len(targets[word]) > 1:
                self.strict_target_words.add(word)

    def remove_punctuation(text):
        return re.sub(r"\p{P}+", " ", text, re.UNICODE)

    def remove_diacritics(text):
        for diac in Evaluator.diac_map:
            text = text.replace(diac, Evaluator.diac_map[diac])
        return text

    def evaluate(self, gold, prediction):
        metrics = {
            "word_all": 0.,
            "word_target": 0.,
            "strict_word_target": 0.,
            "character_all": 0.,
            "character_target": 0.,
            "_word_count": 0,
            "_target_word_count": 0,
            "_strict_target_word_count": 0,
            "_character_count": 0,
            "_target_character_count": 0,
            "error": ""
        }
        try:
            # some type-checking and other initial checks
            if type(gold) is not str or type(prediction) is not str:
                metrics['error'] = f"Gold and/or prediction are not strings ({type(gold)} and {type(prediction)})!"
                return metrics
            if len(gold) == 0 or len(prediction) == 0:
                metrics['error'] = f"Gold and/or prediction strings are empty ({len(gold)} and {len(prediction)})!"
                return metrics
            if len(gold) != len(prediction):
                metrics['error'] = f"Gold and prediction strings have different lengths ({len(gold)}!={len(prediction)})"
                return metrics

            # let's check that we're not going to produce other characters
            position = -1
            for g, p in zip(gold, prediction):
                position += 1
                if g not in Evaluator.target_chars and g != p:
                    metrics['error'] = f"Strings do not match! Character at position {position} in gold is [{g}] (char code {ord(g)}) and in predicted string is [{p}] (char code {ord(p)})!"
                    return metrics

            # now let's check char-level metrics
            count_t, acc_a, acc_t = 0, 0, 0
            for g, p in zip(gold, prediction):
                # all-chars
                if g == p:
                    acc_a += 1
                if g in Evaluator.target_chars:
                    count_t += 1
                    if g == p:
                        acc_t += 1

            metrics['character_all'] = acc_a / len(prediction)  # we're looking at all the string
            metrics['_character_count'] = len(prediction)
            metrics['_target_character_count'] = count_t
            if count_t > 0:
                metrics['character_target'] = acc_t / count_t  # we're only counting our target characters
            else:
                metrics['character_target'] = 0.  # we can't compute anything as we don't have even a single target char

            # finally, word level metrics
            count_t, count_st, word_a, word_t, word_st = 0, 0, 0, 0, 0
            gold = Evaluator.remove_punctuation(gold).strip().replace("  ", " ").split()
            prediction = Evaluator.remove_punctuation(prediction).strip().replace("  ", " ").split()
            for g, p in zip(gold, prediction):
                if g == p:
                    word_a += 1

                g_nodiac = Evaluator.remove_diacritics(g).lower()
                target_word, strict_target_word = False, False
                if g_nodiac in self.strict_target_words:  # a target is always in the strict target, no need to check twice
                    strict_target_word, target_word = True, True
                    count_t += 1
                    count_st += 1
                elif g_nodiac in self.target_words:
                    target_word = True
                    count_t += 1
                else:  # check that gold has at least one diacritic
                    for c in g:
                        if c in Evaluator.diacs:
                            target_word = True
                            count_t += 1
                            count_st += 1
                            break

                if target_word is True and g == p:
                    word_t += 1
                if strict_target_word is True and g == p:
                    word_st += 1

            metrics['word_all'] = word_a / len(prediction)  # we're looking at all the string
            metrics['_word_count'] = len(prediction)
            metrics['_target_word_count'] = count_t
            if count_t > 0:
                metrics['word_target'] = word_t / count_t  # we're only counting our target words
            else:
                metrics['word_target'] = 0.  # we can't compute anything as we don't have even a single target word
            metrics['_strict_target_word_count'] = count_st
            if count_st > 0:
                metrics['strict_word_target'] = word_st / count_st  # we're only counting our target words
            else:
                metrics['strict_word_target'] = 0.  # we can't compute anything as we don't have even a single target word

            return metrics
        except Exception as ex:
            metrics['error'] = f"Exception while computing metrics: {str(ex)}"
            return metrics

# how to use
if __name__ == "__main__":

    # example strings
    gold = "„Fata are un măr în mână."
    prediction = "„Fată are un mar în mâna."

    # init evaluator class
    evaluator = Evaluator()
    metrics = evaluator.evaluate(gold, prediction)

    # print metrics
    if metrics['error'] != "":
        print(f"Error: {metrics['error']}")
    else:
        for key in metrics:
            if type(metrics[key]) is float:
                print(f" {key:>30}: {metrics[key]:.4f}")
            if type(metrics[key]) is int:
                print(f" {key:>30}: {metrics[key]}")
