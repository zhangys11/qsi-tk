'''
Contains Raman prior knowledgebase and functions.
'''
import random
import json
import os.path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import xlrd


def plot_raman_prior(raman_peak_list, group_only=False):
    '''
    plot raman peaks

    Parameters
    ----------
    group_only : whether only plot peaks that are in any group. Otherwise, plot all.
    '''
    plt.figure(figsize=(20, 10))
    colors = list(mcolors.TABLEAU_COLORS.keys())
    dic = {}
    for p in raman_peak_list:
        dic.setdefault((p.chemical, p.vibration), []).append(p)

    # 将字典中键的物质和键值连在一起
    new_dict = {f"{key[0]} {key[1]}": value for key, value in dic.items()}

    # 将上述字典中成组的挑出来
    group_dic = {}
    for key, value in new_dict.items():
        if len(value) >= 2:
            group_dic[key] = value

    if group_only is False:
        for idx, (key, value) in enumerate(new_dict.items()):
            for item in value:
                peak_range = [item.peak_start, item.peak_end]
                _ = plt.hlines(idx, peak_range[0] - 10, peak_range[1] + 10, lw=6, label=key,
                               color=random.choice(colors))
    else:
        for idx, (key, value) in enumerate(group_dic.items()):
            page_range_list = []
            for item in value:
                peak_range = [item.peak_start, item.peak_end]
                page_range_list.append(peak_range)
            color = random.choice(colors)
            for i, _ in enumerate(page_range_list):
                _ = plt.hlines(idx, page_range_list[i][0] - 10, page_range_list[i][1] + 10, lw=6, label=key,
                               color=color)
        # 获取标签列表并去重
        handles, labels = plt.gca().get_legend_handles_labels()
        unique_labels = list(set(labels))
        # 创建新的图例并显示
        new_handles = [handles[labels.index(label)] for label in unique_labels]
        plt.legend(new_handles, unique_labels, bbox_to_anchor=(1.02, 1), loc='upper left', ncol=1, borderaxespad=0.,
                   labelspacing=0.5, handletextpad=0.5, handlelength=2.5)

    plt.yticks([])
    plt.xlabel("Raman shift($cm^{-1}$)", fontsize=18, labelpad=20)
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
        
        # if args are separated params
        if len(args) > 1:
            self.chemical = args[0]
            self.vibration = args[1]
            self.peak_start = float(args[2])
            self.peak_end = float(args[3])
            self.reference = args[4]
            self.comment = args[5]

        # if arg is a dict
        elif isinstance(args[0], dict):
            dic = args[0]

            try:
                self.chemical = dic['chemical']
                self.vibration = dic['vibration']
                self.peak_start = float(dic['peak_start'])
                self.peak_end = float(dic['peak_end'])
                self.reference = dic['reference']
                self.comment = dic['comment']
            except Exception as e:
                print(e)
                print('Skipping this item...')
                print(dic)

    def validate(self):
        '''
        Validate the Raman peak object.
        '''
        return 'peak_start' in self.__dict__ and 'peak_end' in self.__dict__ and self.peak_start >= 0 and self.peak_end >= 0

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

    l = [RamanPeak(dic) for dic in raman_peak_list]
    return list(filter(lambda item: item.validate(), l)) # remove None items

def get_raman_peak_list_from_excel(filepath="raman.xls"):
    '''
    Load the excel file and return a list of Raman peak objects.
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
    
    for i in range(1,raman_peak_sheet.nrows):
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
