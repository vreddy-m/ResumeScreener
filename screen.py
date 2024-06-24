import os, shutil
import fitz 
import docx2txt
import string
import re
import time
import pandas as pd
import datetime
import subprocess
import nltk
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('maxent_ne_chunker', quiet=True)
nltk.download('words', quiet=True)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from striprtf.striprtf import rtf_to_text
from tabulate import tabulate

#Useful Functions
def cleanText(Text):
    '''
    Removes the noise in the text
    '''
    Text = re.sub(' +', ' ', Text)  # remove extra whitespace
    Text = re.sub(r'^https?:\/\/.*[\r\n]*', ' ', Text, flags=re.MULTILINE)  # remove URLs
    Text = re.sub('RT|cc', ' ', Text)  # remove RT and cc
    Text = re.sub('#', '', Text)  # remove hashtags
    Text = re.sub('@', '  ', Text)  # remove mentions
    Text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[]^_`{|}~"""), ' ', Text)  # remove punctuations
    word_tokens = word_tokenize(Text)
    filtered_sentence = [] 
  
    for w in word_tokens: 
        if w not in set(stopwords.words('english')): 
            filtered_sentence.append(w)

    Text = " ".join(filtered_sentence)
    return Text.lower()

def progressBar(iterable, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    total = len(iterable)
    # Progress Bar Printing Function
    def printProgressBar (iteration):
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Update Progress Bar
    for i, item in enumerate(iterable):
        yield item
        printProgressBar(i + 1)
    print()

def similar(document, jd):
    var = [document, jd]
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(var)
    return round((cosine_similarity(count_matrix)[0][1] * 100),2)

def extract_name(lines):
    '''
    Given an input string, returns possible matches for names. Uses regular expression based matching.
    '''

    # Reads Names from the file,
    Names = open("allNames.txt", "r").read().lower()
    # Lookup in a set is much faster
    Names = set(Names.split())
    otherNameHits = []
    nameHits = []
    name = None
    grammar = r'NAME: {<NN.*><NN.*><NN.*>*}'
    # Noun phrase chunk is made out of two or three tags of type NN. (ie NN, NNP etc.)
    chunkParser = nltk.RegexpParser(grammar)
    all_chunked_tokens = []
    for tagged_tokens in lines:
        # Creates a parse tree
        if len(tagged_tokens) == 0: continue # Prevent it from printing warnings
        chunked_tokens = chunkParser.parse(tagged_tokens)
        all_chunked_tokens.append(chunked_tokens)
        for subtree in chunked_tokens.subtrees():
            #  or subtree.label() == 'S' include in if condition if required
            # if subtree.label() == 'NAME':# or subtree.label() == 'S':
            for ind, leaf in enumerate(subtree.leaves()):
                if (leaf[0].lower() in Names):
                    hit = " ".join([el[0] for el in subtree.leaves()[ind:ind+3]])
                    # Check for the presence of commas, colons, digits - usually markers of non-named entities 
                    if re.compile(r'[\d,:]').search(hit): continue
                    nameHits.append(hit)
    # Going for the first name hit
    if len(nameHits) > 0:
        nameHits = [re.sub(r'[^a-zA-Z \-]', '', el).strip() for el in nameHits] 
        name = " ".join([el[0].upper()+el[1:].lower() for el in nameHits[0].split() if len(el)>0])
        otherNameHits = nameHits[1:]
    return name

def extract_email(text):
    emails = re.findall(r"[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+", text)
    try:
        email = emails[0]
    except:
        email = None
    return email


def getPhone(text):
    '''
    Given an input string, returns possible matches for phone numbers. Uses regular expression based matching.
    '''
    number = None
    try:
        text = re.sub(r'^http?:\/\/.*[\r\n]*', ' ', text, flags=re.MULTILINE)  # remove URLs
        text = re.sub(r"[a-z0-9]+/[0-9]+", ' ', text, flags=re.MULTILINE)
        pattern = re.compile(r'([+(]?\d+[)\-]?[ \t\r\f\v]*[(]?\d{2,}[()\-]?[ \t\r\f\v]*\d{2,}[()\-]?[ \t\r\f\v]*\d*[ \t\r\f\v]*\d*[ \t\r\f\v]*)')
        match = pattern.findall(text)
        match = [re.sub(r'[,.]', '', el) for el in match if len(re.sub(r'[()\-.,\s+]', '', el))>7]
            # Taking care of years, eg. 2001-2004 etc.
        match = [re.sub(r'\D$', '', el).strip() for el in match]
            # matches end of string. This takes care of random trailing non-digit characters. \D is non-digit characters
        match = [el for el in match if len(re.sub(r'\D','',el)) <= 15]
            # Remove number strings that are greater than 15 digits
        try:
            for el in list(match):
                # Create a copy of the list since you're iterating over it
                if len(el.split('-')) > 3: continue # Year format YYYY-MM-DD
                for x in el.split("-"):
                    try:
                        if x.strip()[-4:].isdigit():
                            if int(x.strip()[-4:]) in range(1900, 2100):
                                # Don't combine the two if statements to avoid a type conversion error
                                match.remove(el)
                    except:
                        pass
        except:
            pass
        number = set(match)
    except:
        pass
    return number


def experience(lines):
    for sentence in lines:
        sen = ' '.join([words[0].lower() for words in sentence])
        if re.search('experience', sen):
            sen_tokenised = nltk.word_tokenize(sen)
            tagged = nltk.pos_tag(sen_tokenised)
            entities = nltk.chunk.ne_chunk(tagged)
            for subtree in entities.subtrees():
                for leaf in subtree.leaves():
                    if leaf[1] == 'CD':
                        experience = leaf[0]
                        return experience


# Getting the job titles
df=pd.read_csv(r"jd.csv", usecols=['Job'])
df_dic = df.to_dict()
if df_dic.get('Job'):
    print('\n')
    for key, value in df_dic.get('Job').items():
        print(f" select '{key}' for {value}")
    print('\n')
    role = input('Please choose a number from the above list : ')
    try:
        df_role = df_dic['Job'][int(role)]
        df=pd.read_csv(r"jd.csv")
        df_jd = df[df['Job'] == df_role]
        index = df_jd['JD'].keys()[0]
        job_des = cleanText(df_jd['JD'][index])

        # removing repeated words
        final_jd = ' '.join(dict.fromkeys(job_des.split()))
        
        # Resumes
        lst = []
        path = f'{os.getcwd()}/Resumes/'
        if len(os.listdir(path)) > 0:
            for filename in progressBar(os.listdir(path), prefix = 'Scanning:', suffix = 'Completed', length = 50):
                filepath = path + filename
                dic = {}
                if filepath.endswith(".docx"):
                    txt = docx2txt.process(filepath)
                    if txt:
                        lines = [item.strip() for item in txt.split('\n') if len(item) > 0]
                        lines = [nltk.word_tokenize(item) for item in lines]
                        lines = [nltk.pos_tag(item) for item in lines]
                        doc_txt = cleanText(txt.replace('\t', ' '))
                        response = similar(doc_txt, final_jd)
                        exp = experience(lines)
                        name = extract_name(lines)
                        email = extract_email(txt)
                        phone = getPhone(txt)
                        dic['score'] = response
                        dic['name'] = name
                        dic['exp'] = exp
                        dic['email'] = email
                        dic['phone'] = phone
                        dic['resume'] = filename
                        lst.append(dic)
                
                elif filepath.endswith(".pdf"):
                    try:
                        with fitz.open(filepath) as doc:
                            text = ""
                            for page in doc:
                                text += page.getText()
                            lines = [item.strip() for item in text.split('\n') if len(item) > 0]
                            lines = [nltk.word_tokenize(item) for item in lines]
                            lines = [nltk.pos_tag(item) for item in lines]
                            pdf_txt = cleanText(text)
                            response = similar(pdf_txt, final_jd)
                            exp = experience(lines)
                            name = extract_name(lines)
                            email = extract_email(text)
                            phone = getPhone(text)
                            dic['score'] = response
                            dic['name'] = name
                            dic['exp'] = exp
                            dic['email'] = email
                            dic['phone'] = phone
                            dic['resume'] = filename
                            lst.append(dic)
                    except:
                        pass
                elif filepath.endswith(".doc"):
                    extension = 'doc'
                    txt = subprocess.Popen(['antiword', filepath], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True).communicate()[0], extension
                    try:
                        txt = txt[0].decode("utf-8")
                    except:
                        txt = txt[0].decode("latin-1")
                    txt = ''.join(txt)
                    if txt:
                        lines = [item.strip() for item in txt.split('\n') if len(item) > 0]
                        lines = [nltk.word_tokenize(item) for item in lines]
                        lines = [nltk.pos_tag(item) for item in lines]
                        doc_txt = cleanText(txt.replace('\t', ' '))
                        response = similar(doc_txt, final_jd)
                        exp = experience(lines)
                        name = extract_name(lines)
                        email = extract_email(txt)
                        phone = getPhone(txt)
                        dic['score'] = response
                        dic['name'] = name
                        dic['exp'] = exp
                        dic['email'] = email
                        dic['phone'] = phone
                        dic['resume'] = filename
                        lst.append(dic)
                elif filepath.endswith(".rtf"):
                    txt = open(filepath).read()
                    txt = rtf_to_text(txt)
                    if txt:
                        lines = [item.strip() for item in txt.split('\n') if len(item) > 0]
                        lines = [nltk.word_tokenize(item) for item in lines]
                        lines = [nltk.pos_tag(item) for item in lines]
                        doc_txt = cleanText(txt.replace('\t', ' '))
                        response = similar(doc_txt, final_jd)
                        exp = experience(lines)
                        name = extract_name(lines)
                        email = extract_email(txt)
                        phone = getPhone(txt)
                        dic['score'] = response
                        dic['name'] = name
                        dic['exp'] = exp
                        dic['email'] = email
                        dic['phone'] = phone
                        dic['resume'] = filename
                        lst.append(dic)
                time.sleep(0.1)

            # name of the file
            current_time = str(datetime.datetime.now())
            file_name = f'{df_role}_{current_time[:10]}_{current_time[11:13]}_{current_time[14:16]}.xlsx'
            
            # saving the excel
            for i in lst:
                new = ', '.join(i['phone'])
                i['phone'] = new
                i['current_location'] = None
                i['cctc'] = None
                i['ectc'] = None
            df = pd.DataFrame(sorted(lst, key = lambda i: i['score'], reverse=True)) 
            df.to_excel(file_name)
            
            # for displaying top 5
            df = df.drop(['resume','current_location', 'cctc', 'ectc'], axis=1)
            df.index = df.index + 1
            print('\n')
            print('Top 5 profiles are shown below :')
            print('\n')
            print(tabulate(df.head(5), headers = 'keys', tablefmt = 'psql'))
            
            #Creating new folder and copying files to it.
            new_folder_name = f'{str(df_role)}_{current_time[:10]}_{current_time[11:13]}_{current_time[14:16]}'
            new_dir = os.path.join(os.getcwd(), new_folder_name)
            os.mkdir(new_dir)
            files_list = [r['resume'] for r in sorted(lst, key = lambda i: i['score'], reverse=True)][:10]
            for f in files_list:
                filepath = path + f
                shutil.copy(filepath, new_dir)
            print(f'Profiles are written to "{file_name}" sheet and copied to a new folder named {new_folder_name}')
            print('\n')
        else:
            print('Resumes folder is empty, Please add the resumes in the folder')
    except:
        print('Please Enter a Valid Number from the above list')
else:
    print('Job description sheet is empty')