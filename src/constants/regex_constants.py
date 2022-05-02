# Date finding
RAW_MM_DD_YYYY = "(\\d\\d)/(\\d\\d)/(\\d\\d(?:\\d\\d)?)"

MM_SLASHORONE_DD_SLASH_YYYY = "([01]?\\d)[/1](\\d?\\d)/(\\d\\d(?:\\d\\d)?)"
MM_SLASH_DD_ONE_YYYY = "([01]?\\d)/(\\d?\\d)1(\\d\\d(?:\\d\\d)?)"
MM_DASH_DD_DASH_YYYY = "([01]?\\d)-(\\d?\\d)-(\\d\\d(?:\\d\\d)?)"

ADMISSION_OR_SERVICE_DATE_MMDDYYYY = "(?:Adm(?:ission)? Date|Date of Service):\\s*(\\d\\d)/(\\d\\d)/(\\d\\d(?:\\d\\d)?)"

# Document Splitting
PAGE_NUMBER_WITH_MAX = '[Pp]age:? (\\d+) of (\\d+)'
BARE_PAGE_NUMBER = '[Pp]age:? (\\d+)(?: of (\\d+))?'

# A series of cases for what constitutes the end of a sentence.
# 1. ! and ? are easy first case.
# 2. Avoid breaking on periods when there's a title or numbered list involved
# 3. Common list formats
SENTENCE_REGEX = "(?s)\\S.*?([!?](?=\\s)|" + \
                 "(?<!\\n\\d|( |\\.)[a-zA-Z]|Mr|Dr|Ms|vs)(?<!Mrs)(?<!^\\d)\\.(?=\\s)|" + \
                 "(?=\\s*?\\n\\d\\.\\s|\\s*?\\n\\s|(?<![: ]) 5,| 6,|\\t2,| 2,\\t+|\\t+\\s)|$)"

# Tokenization regex
ICD10_CODE = "([a-zA-Z]\\d\\w\\.\\w{1,4})"
LONG_NUMBER = "\\d{11,}"
PERSONAL_TITLE = "(([DdMm]r)|([Mm]r?s)\\.)"
CONTRACTION = "(\\w+'\\w{1,2})"
INITIAL = "([a-zA-Z]\\.)+"
DOT_JOINED_NUMBER = "(\\d*\\.\\d+)"
UNIT_WITH_SLASH = "g/dl|g/dL|G/DL|" + \
                  "MG/DL|mg/dl|mg/dL|" + \
                  "IU/mL|iu/ml|IU/ML|" + \
                  "IU/L|iu/l|iu/L|" + \
                  "mmol/L|mmol/l|MMOL/L|" + \
                  "ug/mL|ug/ml|UG/ML|" + \
                  "K/uL|k/ul|K/UL|" + \
                  "pg/mL|pg/ml|PG/ML|" + \
                  "u/ml|u/mL|U/ML|" + \
                  "(?:Units|UNITS|units)/(?:g|G|mL|ml|ML)"

YEARS_REGEX = "19[4-9]\\d|20\\d{2}"
MONTHS_REGEX = "(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul" + \
               "(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|(Nov|Dec)(?:ember)?)"
DATETIME_REGEX = "(\\s[0-2][0-9]\\:[0-5][0-9]\\:[0-5][0-9])?"
TRIGRAM_QUADGRAM_DATE_DATETIME = "\\b([0-3]?[0-9](st|nd|rd|th)" + \
                                 "+|" + YEARS_REGEX + "|[0-3]?\\d{1}|" + MONTHS_REGEX + ")+([,.\\-\\/\\ ])+" + \
                                 "(\\b[0-3]?\\d{1}|" + MONTHS_REGEX + ")+" + \
                                 "([,.\\-\\/\\ ])+([0-2][0-9]\\:[0-5][0-9]\\:[0-5][0-9]\\s)?" + \
                                 "(" + YEARS_REGEX + "|\\d{1,2})?" + DATETIME_REGEX + "(?<! )\\b"
BIGRAM_FOUR_DIGIT_YEAR_MONTH = "\\b((" + YEARS_REGEX + ")\\-\\d{1,2})"
BIGRAM_TEXT_MONTH_YEAR = "\\b" + MONTHS_REGEX + "+\\s(" + YEARS_REGEX + ")+\\b"
BIGRAM_TEXT_YEAR_MONTH = "\\b(" + YEARS_REGEX + "\\s)+" + MONTHS_REGEX + "+\\b"
BIGRAM_TEXT_MONTH_DAY = "([0-3]?[0-9](st|nd|rd|th)?\\s)?" + MONTHS_REGEX + "+(\\s[0-3]?[0-9](st|nd|rd|th)?)?\\b"
UNIGRAM_MONTH_YEAR = "\\b" + MONTHS_REGEX + "+(" + YEARS_REGEX + ")+\\b"
UNIGRAM_MONTH = "\\b" + MONTHS_REGEX + "+\\b"
UNIGRAM_YEAR = "\\b(" + YEARS_REGEX + ")+\\b"
DATE = "((0?[1-9]|1[012])[-/](0?[1-9]|[12]\\d|3[01])[-/](19|20)?\\d\\d)"
TIME_12HR = "((1[012]|[1-9]):[0-5]\\d(\\s)?([AaPp][Mm]))"
TIME_24HR = "(([01]?\\d|2[0-3]):[0-5]\\d(:[0-5]\\d)?)"
HYPHENATED_TOKEN = "(\\w+-\\w+)"
NUMBER = "\\d+(?:\\.\\d+)?"

# Value Extraction regex

MED_FREQUENCY = "QD|qd|BID|bid|TID|tid|QID|qid|PRN|prn|QOD|qod|q\\d+h|Q\\d+H|Q\\d+h"
MED_ROUTE = "IV|iv|PO|po|IM|im|IO|io|GTT|gtt"
MED_FORMULA = "TABLET|TAB|tablet|tab|elixer|ELIXER|Elixer|ELIX|elix|PILL|pill|CAP|cap|CAPSULE|cream|CREAM|" \
              "paste|PASTE|TAB.SR|tab.sr|AMPUL"
BASIC_UNIT = "(?:[kKcCdDMm]|MC|Mc|mc|mC)?[gGcCmMfFSs]|" \
             "mol|MOL|inch|INCH|in|IN|inches|INCHES|oz|OZ|\"|'|%|feet|FEET|ft|FT|bpm|BPM|ml|ML|mL|" \
             "pounds?|POUNDS?|ounces?|OUNCES?|lbs?|LBS?|Units|UNITS|units"
UNIT = "(" + BASIC_UNIT + "|" + UNIT_WITH_SLASH + ")"
NUMBER_AND_UNIT = "(" + NUMBER + ")(\\s?)(?:" + UNIT + ")"
RANGE_AND_UNIT = "(" + NUMBER + ")(\\s?(?:-|~)\\s?)(" + NUMBER + ")(\\s?)(?:" + UNIT + ")?"
HEIGHT = "(" + NUMBER + ")( ?)(feet|FEET|ft|FT|foot|FOOT|')(\\.?,?\\s*)" \
                        "(" + NUMBER + ")((?:[- ])?(\\d/\\d))?( ?)(inch|INCH|in\\.?|IN\\.?|inches|INCHES|\")?"
NUMBER_RANGE_AND_UNIT = "(" + NUMBER + ")(\\s+)" + "(" + NUMBER + ")" + "(\\s*-*\\s*)" \
                                                                        "(" + NUMBER + ")(\\s?)(?:" + UNIT + ")"
WEIGHT = "(" + NUMBER + ")(\\s*)(LBS?|lbs?|kg|KG|pounds|POUNDS)(\\s*)(" + NUMBER + ")(\\s*)(oz|OZ|g|G|ounces|OUNCES)?"

# Patient Identification

# Shapes of names
FIRST_NAME = '[A-Za-z]{2,}'
LAST_NAME = '[A-Za-z\\-\']{2,}'
FIRST_NAME_LAST_NAME = '('+FIRST_NAME+')(?:[ \\t]+\\w\\.?)?[ \\t]+('+LAST_NAME+')'
FIRST_MIDDLE_LAST_NAME = '('+FIRST_NAME+')[ \\t]+('+FIRST_NAME+')[ \\t]+('+LAST_NAME+')'
LAST_NAME_COMMA_FIRST_NAME = '('+LAST_NAME+')(?: JR\\.?)?[,.][ \\t]*('+FIRST_NAME+')'
LAST_NAME_COMMA_FIRST_MIDDLE = '('+LAST_NAME+')(?: JR\\.?)?[,.][ \\t]*('+FIRST_NAME+')[ \\t]+('+FIRST_NAME+')'

# Contexts that appear before/after names in the text
NARROW_PATIENT_NAME_PREFIX = \
    '\\b(?:PATIENT|MEMBER|PT|GUARANTOR|SUBSCRIBER|CARRIER|INSURED)(?:\\s+NAME\\s*[:;.]?|\\s*[:;])\\s*'
BROAD_PATIENT_NAME_PREFIX = \
    '(?:PATIENT|MEMBER|PT|GUARANTOR|SUBSCRIBER|CARRIER|NAME|RE|REGARDING)(?: NAME\\s*[:;.]?|\\s*[:;.])\\s*'
PATIENT_NAME_SUFFIX = '[ \\t]+\\(?(?:MRN|DOB)'

# Patterns for contexts like "Patient Name: John Doe"
FIRST_LAST_NARROW = NARROW_PATIENT_NAME_PREFIX + FIRST_NAME_LAST_NAME
FIRST_MIDDLE_LAST_NARROW = NARROW_PATIENT_NAME_PREFIX + FIRST_MIDDLE_LAST_NAME
LAST_FIRST_NARROW = NARROW_PATIENT_NAME_PREFIX + LAST_NAME_COMMA_FIRST_NAME

FIRST_LAST_BROAD = BROAD_PATIENT_NAME_PREFIX + FIRST_NAME_LAST_NAME
FIRST_MIDDLE_LAST_BROAD = BROAD_PATIENT_NAME_PREFIX + FIRST_MIDDLE_LAST_NAME
LAST_FIRST_BROAD = BROAD_PATIENT_NAME_PREFIX + LAST_NAME_COMMA_FIRST_NAME

# Patterns for contexts like "John Doe (MRN: ..."
FIRST_LAST_SUFFIX = FIRST_NAME_LAST_NAME + PATIENT_NAME_SUFFIX
FIRST_MIDDLE_LAST_SUFFIX = FIRST_MIDDLE_LAST_NAME + PATIENT_NAME_SUFFIX
LAST_FIRST_SUFFIX = LAST_NAME_COMMA_FIRST_NAME + PATIENT_NAME_SUFFIX
LAST_FIRST_MIDDLE_SUFFIX = LAST_NAME_COMMA_FIRST_MIDDLE + PATIENT_NAME_SUFFIX

LONE_FIRST_NAME = 'FIRST(?: ?NAME)?[:.]\\s*([A-Za-z]{2,})'
LONE_LAST_NAME = 'LAST(?: ?NAME)?[:.]\\s*([A-Za-z\\-]{2,})'

# Patterns to try and extract DOB and Patient ID
DOB_PREFIX = '(?:DATE OF BIRTH|BIRTH ?DATE|DOB|D\\.O\\.B\\.)\\s*[:;.]*\\s*'
MM_DD_YYYY = '(\\d{1,2})[/\\-\\s](\\d{1,2})[/\\-\\s](\\d\\d(?:\\d\\d)?)'
DATE_OF_BIRTH = DOB_PREFIX + MM_DD_YYYY

MONTHS_REGEX = "(?:JAN(?:UARY)?|FEB(?:RUARY)?|MAR(?:CH)?|APR(?:IL)?|MAY|JUN(?:E)?|JUL(?:Y)?|" \
               "AUG(?:UST)?|SEP(?:TEMBER)?|OCT(?:OBER)?|(?:NOV|DEC)(?:EMBER)?)"
MONTH_MAPPING = {'JAN': '01', 'FEB': '02', 'MAR': '03', 'APR': '04', 'MAY': '05', 'JUN': '06',
                 'JUL': '07', 'AUG': '08', 'SEP': '09', 'OCT': '10', 'NOV': '11', 'DEC': '12'}
LONGFORM_DATE = '(' + MONTHS_REGEX + ')\\s+(\\d{1,2})(?:ST|ND|RD|TH)? ?[,.]?\\s*(\\d\\d(?:\\d\\d)?)'
LONGFORM_DOB = DOB_PREFIX + LONGFORM_DATE

PATIENT_ID = '(?:PATIENT|MEMBER) ?(?:ID|NUMBER)\\s*:?\\s*(\\w+)'
