import matplotlib.pyplot as plt

def raman_prior():
    '''
    Return Raman prior knowledge, i.e., what wavenumber ranges correspond to what functional groups (chemical bonds).
    TODO: This function is now obsolete. Need update.
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

def plot_raman_prior():

    d = raman_prior()

    plt.figure(figsize = (14,7))

    for idx, key in enumerate(d):
        # print(d[key])
        plt.scatter(d[key], [-idx] * len(d[key]), lw = 5, label = key)
        
    plt.legend(loc = "upper right")
    plt.yticks([])
    plt.xticks(range(500, 3001, 500))
    plt.show()


class RamanPeak:

    def __init__(self,chemical='',vibration='',peak_start=0,peak_end=0,reference='',comment=''):
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
        self.chemical = chemical
        self.vibration = vibration 
        self.peak_start = peak_start
        self.peak_end = peak_end
        self.reference = reference
        self.comment = comment

    def __str__(self):
        return f'{self.chemical} {self.vibration} {self.peak_start} {self.peak_end} {self.reference} {self.comment}'
    
    def __repr__(self):
        return f'{self.chemical} {self.vibration} {self.peak_start} {self.peak_end} {self.reference} {self.comment}'    
    
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
    '''

    json.dupm(raman_peak_list, filepath)

def load_raman_peak_list(filepath):
    '''
    Load the list of Raman peak objects from a file.
    '''

    return json.load(filepath)

def get_raman_peak_list_from_excel(filepath):
    '''
    Load the excel file and return a list of Raman peak objects.
    '''

    raman_peak_list = []

    # ...
    return raman_peak_list

def generate_html_table(raman_peak_list):
    '''
    Generate HTML table from the list of Raman peak objects.
    '''

    html = '<table border="1">'
    html += '<tr><th>Chemical</th><th>Vibration</th><th>Peak Start</th><th>Peak End</th><th>Reference</th><th>Comment</th></tr>'
    for raman_peak in raman_peak_list:
        html += raman_peak.__html__()
    html += '</table>'

    return html