import json
import os.path
import matplotlib.pyplot as plt
import xlrd
import matplotlib.colors as mcolors
import random

def raman_prior_sample():
    '''
    Return Raman prior knowledge, i.e., what wavenumber ranges correspond to what functional groups (chemical bonds).
    This function is now obsolete. Just used to generate sample data.
    '''

    d = {}

    d['alkane_1'] = list(range(1295, 1305 + 1))
    d['alkane_2'] = list(range(800, 900 + 1)) + list(range(1040, 1100 + 1))
    d['branched_alkane_1'] = list(range(900, 950 + 1))
    d['branched_alkane_2'] = list(range(1040, 1060 + 1))
    d['branched_alkane_3'] = list(range(1140, 1170 + 1))
    d['branched_alkane_4'] = list(range(1165, 1175 + 1))
    d['haloalkane_1'] =  list(range(605, 615 + 1))
    d['haloalkane_2'] =  list(range(630, 635 + 1))
    d['haloalkane_3'] =  list(range(655, 675 + 1))
    d['haloalkane_4'] =  list(range(740, 760 + 1))
    d['alkene'] = list(range(1638, 1650 + 1))
    d['alkyne'] = list(range(2230, 2237 + 1))
    d['toluence'] = list(range(990, 1010 + 1))
    d['alcohol'] = list(range(800, 900 + 1))
    d['aldehyde'] = list(range(1725, 1740 + 1))
    d['ketone'] = list(range(1712, 1720 + 1))
    d['ether'] = list(range(820, 890 + 1))
    d['carboxylic_acid'] = list(range(820, 890 + 1))
    d['ester'] = list(range(634, 644 + 1))
    d['amine_1'] = list(range(740, 833 + 1))
    d['amine_2'] = list(range(1000, 1250 + 1))
    d['amide'] = list(range(700, 750 + 1))
    d['nitrile'] = list(range(2230, 2250 + 1))

    return d

def plot_raman_prior(raman_peak_list):
    '''
    plot raman peaks
    '''

    plt.figure(figsize = (10, 5))
    colors=list(mcolors.TABLEAU_COLORS.keys())

    dic = {}
    for p in raman_peak_list:
        dic.setdefault((p.chemical, p.vibration),[]).append(p)
        
    for idx, (key, value) in enumerate(dic.items()):
        for item in value:
            peak_range=[item.peak_start, item.peak_end]
        _ = plt.hlines(idx,peak_range[0]-10,peak_range[1]+10, lw = 3, label = key,color=random.choice(colors))
        
    # plt.legend(loc = "lower right")
    plt.yticks([])
    plt.xticks([])
    plt.show()

class RamanPeak:

    '''
    def __init__(self,chemical='',vibration='',peak_start=0,peak_end=0,reference='',comment=''):

        self.chemical = chemical
        self.vibration = vibration 
        self.peak_start = peak_start
        self.peak_end = peak_end
        self.reference = reference
        self.comment = comment
    '''

    def __init__(self, *args):
        '''
        Parameters
        ----------
        chemical : Chemical name.
        vibration : Function groups or chemical bond and their vibration modes.
        peak_start : Start of the peak.
        peak_end : End of the peak.
        reference : URL or DOI.
        comment : Comment.
        '''
        
        # if args are more than 1 sum of args
        if len(args) > 1:
            self.chemical = args[0]
            self.vibration = args[1]
            self.peak_start = float(args[2])
            self.peak_end = float(args[3])
            self.reference = args[4]
            self.comment = args[5]

        # if arg is an integer square the arg
        elif isinstance(args[0], dict):
            dic = args[0]
            self.chemical = dic['chemical']
            self.vibration = dic['vibration'] 
            self.peak_start = float(dic['peak_start'])
            self.peak_end = float(dic['peak_end'])
            self.reference = dic['reference']
            self.comment = dic['comment']

    def __str__(self):
        return f'{self.chemical} {self.vibration} {self.peak_start} {self.peak_end} {self.reference} {self.comment}'
    
    def __repr__(self):
        return f'{self.chemical} {self.vibration} {self.peak_start} {self.peak_end} {self.reference} {self.comment}'    
    
    def __enter__(self):
        pass
    
    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def is_same_group(self, other):
        '''
        Return True if two Raman peaks are in the same group.
        ''' 
        return self.chemical == other.chemical and self.vibration == other.vibration 
    
    def __html__(self):
        return '<tr><td>' + self.chemical + '</td><td>' + self.vibration + '</td><td>' + str(self.peak_start) + '</td><td>' + str(self.peak_end) + '</td><td>' + self.reference + '</td><td>' + self.comment + '</td></tr>'
    

def save_raman_peak_list(raman_peak_list, filepath):
    '''
    Save the list of Raman peak objects to a file.

    Note
    ----
    Use the following code to dump the Raman info from excel to a json file.
        raman_peak_list = get_raman_peak_list_from_excel()
        save_raman_peak_list(save_raman_peak_list)
    '''
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(raman_peak_list, f, indent=4)


def load_raman_peak_list(json_file = None):
    '''
    Load the list of Raman peak objects from a file.
    '''

    if json_file is None:
        json_file = os.path.realpath(__file__).replace('.py', '.json')

    with open(json_file, encoding="utf-8") as f:
        raman_peak_list = json.load(f)

    return [RamanPeak(dic) for dic in raman_peak_list]

def get_raman_peak_list_from_excel(n = 194, filepath="D:\\group lasso\\raman.xls"):
    '''
    Load the excel file and return a list of Raman peak objects.

    TODO: n : get from excel rows.
    '''
    raman_peak_excel = xlrd.open_workbook(filepath)#括号里为路径
    raman_peak_sheet = raman_peak_excel.sheet_by_index(0)#索引至页
   
    #中间量，以暂时存储从表中读取的数据
    chemicals=[]
    vibrations=[]
    peak_starts=[]
    peak_ends=[]
    references=[]
    comments=[]    
    
    for i in range(1,n+1): 
        chemicals.append(raman_peak_sheet.cell_value(i,0))
        vibrations.append(raman_peak_sheet.cell_value(i,1))
        peak_starts.append(raman_peak_sheet.cell_value(i,2))
        peak_ends.append(raman_peak_sheet.cell_value(i,3))
        references.append(raman_peak_sheet.cell_value(i,4))
        comments.append(raman_peak_sheet.cell_value(i,5))

    raman_peak_list = [{'chemical': c, 'vibration': v, 'peak_start': ps, 'peak_end': pe, 'reference': r, 'comment': co} for c, v, ps, pe, r, co in zip(chemicals, vibrations, peak_starts, peak_ends, references, comments)]
    return raman_peak_list


def generate_html_table(raman_peak_list):
    '''
    Generate HTML table from the list of Raman peak objects.

    Example
    -------
    raman_peak_list = load_raman_peak_list()
    html = generate_html_table(raman_peak_list)
    IPython.display.display(IPython.display.HTML(html))
    '''
    # ranman_html = pd.DataFrame.from_dict(raman_peak_list)
    # html_text = ranman_html.to_html(justify='center')
    # print(html_text)

    html = '<table border="1">'
    html += '<tr><th>Chemical</th><th>Vibration</th><th>Peak Start</th><th>Peak End</th><th>Reference</th><th>Comment</th></tr>'
    for raman_peak in raman_peak_list:
        html += raman_peak.__html__()
    html += '</table>'

    return html