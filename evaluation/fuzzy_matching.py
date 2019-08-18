#--*-- encoding:utf-8 --*--
import sys
import re
import codecs
from collections import defaultdict
from state_matcher import StateMatcher
from names import country_and_region_names as crnames
from names import chemical_element_names as cenames
from names import measure_names
from re_util import build_greedy_regex_str

__all__ = ['FuzzyMatcher']

DEBUG=False

class FuzzyMatcher(object):
    def __init__(self):
        self.__PATTERN_SPACE = re.compile(r'\s+', re.UNICODE)
        self.synsets = defaultdict(lambda: set())
        self.reverse_synsets = defaultdict(lambda: [])

        from synsets import synsets as synsets_
        for i, synset in enumerate(synsets_):
            for word in synset:
                synset = self.__key(word)
                self.synsets[word].add(i)
            self.reverse_synsets[i] = synset
        del synsets_

        self.__build_patterns()
        self.state_matcher = StateMatcher()

    def __key(self, text):
        if not isinstance(text, str):
            text = text.decode('utf-8')
        return self.__PATTERN_SPACE.sub('', text).lower()

    def is_synonym(self, std_text, other_text):
        std_text_ori, other_text_ori = std_text, other_text
        std_text, other_text = self.__key(std_text), self.__key(other_text)

        # if DEBUG: print(std_text.encode('utf-8'), ' ||| ', end=' ', file=sys.stderr)
        # if DEBUG: print(other_text.encode('utf-8'), file=sys.stderr)

        # equal
        if std_text == other_text: return True
        # if DEBUG: print('\tnot equal', file=sys.stderr)

        # in synonym list
        synset1 = self.synsets[std_text]
        synset2 = self.synsets[other_text]
        if len(synset1 & synset2) != 0:
            return True
        # if DEBUG: print('\tnot synset', file=sys.stderr)

        # other rules
        if self.check_prefix_and_suffix(std_text, other_text):
            return True
        # if DEBUG: print('\tnot other rules', file=sys.stderr)

        # state names
        if self.check_state_names(std_text_ori, other_text_ori):
            return True
        # if DEBUG: print('\tnot state names', file=sys.stderr)

        # if DEBUG: print('\tOK, not synonym ', file=sys.stderr)
        return False

    __PATTERN_STATE_NAME = re.compile('^(.+)((自治)?(市|县|旗|区|盟|州)|特别行政区)$')
    def generate_candidate_synonym(self, std_text):
        if not isinstance(std_text, str):
            std_text = std_text.decode('utf-8')

        candidates = set()
        candidates.add(std_text)

        # clean up
        std_text = std_text.replace(' ', '').lower()
        std_text = std_text.lstrip('"').lstrip('“').lstrip("'")\
                           .lstrip("‘").lstrip('<').lstrip('《')\
                           .lstrip('【')
        std_text = std_text.rstrip('"').rstrip('”').rstrip("'")\
                           .rstrip("’").rstrip('>').rstrip('》')\
                           .rstrip("】")
        candidates.add(std_text)

        # synonyms
        candidates.update(self.reverse_synsets[std_text])

        # dynasty
        if std_text.endswith('朝') and self.is_synonym(std_text, std_text[0:-1]):
            candidates.add(std_text[0:-1])

        # state names
        m = self.__PATTERN_STATE_NAME.match(std_text)
        if m and self.is_synonym(std_text, m.group(1)):
            candidates.add(m.group(1))

        return candidates

    def check_state_names(self, std_text, other_text):
        return self.state_matcher.is_same_complex_state(std_text, other_text)

    def __build_patterns(self):
        measures = '|'.join(measure_names)
        digits = r'〇一二三四五六七八九十百千万亿兆\d'
        country_and_region_names = build_greedy_regex_str(crnames)
        chemical_element_names = build_greedy_regex_str(cenames)
        pattern_pairs1 = [ # short, long
                    (r'^([{digits}]+)#第?\1({measures})$'.format(digits=digits, measures=measures)),
                    (r'^([{digits}]+({measures}))#第\1$'.format(digits=digits, measures=measures)),
                    (r'^([{digits}]+)#\1[岁]$'.format(digits=digits)),
                    (r'^([{digits}]+个)#\1[人月]$'.format(digits=digits)),
                    (r'^(十一|十二|[〇一二三四五六七八九十]|\d+)月?#\1月份$'),
                    (r'^([{digits}]+)年?#公元前?\1年?$'.format(digits=digits)),
                    (r'^([{digits}]+(分|秒))#\1钟$'.format(digits=digits)),
                    (r'^[《<]?<?(.+)>?[>》]#[《<]<?\1>?[>》]$'),
                    (r'^[《<]?<?(.+)>?[>》]?#[《<]?<?\1>?[>》]$'),
                    (r'^[《<]?<?(.+)>?[>》]?#[《<]<?\1>?[>》]?$'),
                    (r'^(..)#\1座$'), # constellation
                    (r'^(.+国)#\1人$'),
                    (r'^({names})#\1人$'.format(names=country_and_region_names)),
                    (r'^([男女])#\1性$'),
                    (r'^(.+)#\1地区$'),
                    (r'^(.+)#\1(族|民族)$'),
                    (r'^(.+)#\1色$'),
                    (r'^(夏|商|周|秦|汉|晋|南北|南|北|隋|唐|宋|金|元|明|清)#\1朝$'),
                    (r'^(正比|反比)#成\1$'),
                    (r'^(正比|反比)#\1例$'),
                    (r'^({names})#\1元素$'.format(names=chemical_element_names)),
                ]

        pattern_pairs2 = [ # std, other
                    (r'^(.+)的#\1$'),
                    (r'^([鼠|牛|虎|兔|龙|蛇|马|羊|猴|鸡|狗|猪])年#\1$'),
                    (r'^(.+)#\1(公司|大学|罪)$'),
                    (r'^[“‘\'"](.+)[”’\'"]#\1$'),
                    (r'^(.+)#\1国$'),
                ]

        for i in range(len(pattern_pairs1)):
            pattern_pairs1[i] = re.compile(pattern_pairs1[i], re.UNICODE)

        for i in range(len(pattern_pairs2)):
            pattern_pairs2[i] = re.compile(pattern_pairs2[i], re.UNICODE)

        self.pattern_pairs1 = pattern_pairs1
        self.pattern_pairs2 = pattern_pairs2

    def check_prefix_and_suffix(self, std_text, other_text):
        if len(std_text) < len(other_text):
            sstr, lstr = std_text, other_text
        else:
            sstr, lstr = other_text, std_text

        pos = lstr.find(sstr)
        if pos < 0: return False

        for i, p in enumerate(self.pattern_pairs1):
            if p.match(sstr+'#'+lstr):
                # if DEBUG: print('\trule set 1 #%d matched' % i, file=sys.stderr)
                return True

        for i, p in enumerate(self.pattern_pairs2):
            if p.match(std_text+'#'+other_text):
                # if DEBUG: print('\trule set 2 #%d matched' % i, file=sys.stderr)
                # print(s, end="", file=depend)
                return True

        return False
