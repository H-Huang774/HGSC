import subprocess
import time
import os; rootdir = os.path.split(__file__)[0]




def gpcc_coord_encode(filedir, bin_dir, posQuantscale=1, tmc3dir='tmc3_v23', cfgdir='cfgs/ford_01_q1mm', DBG=False):
    tmc3dir = os.path.join(rootdir, tmc3dir)
    cfgdir = os.path.join(rootdir, cfgdir)
    cmd = tmc3dir + ' --mode=0 ' \
        + ' --config='+cfgdir \
        + ' --positionQuantizationScale='+str(posQuantscale) \
        + ' --uncompressedDataPath='+filedir \
        + ' --compressedStreamPath='+bin_dir
    # cmd += '> nul 2>&1' if os.name == 'nt' else '> /dev/null 2>&1'
    subp = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    subp.wait()
    if DBG: print_log(subp)

    return subp

def gpcc_coord_decode(bin_dir, dec_dir,  tmc3dir='tmc3_v23', cfgdir='cfgs/ford_01_q1mm', DBG=False):
    tmc3dir = os.path.join(rootdir, tmc3dir)
    cfgdir = os.path.join(rootdir, cfgdir)
    cmd = tmc3dir + ' --mode=1 ' \
        + ' --config='+cfgdir \
        + ' --compressedStreamPath='+bin_dir \
        + ' --reconstructedDataPath='+dec_dir \
        + ' --outputBinaryPly=0'
    # cmd += '> nul 2>&1' if os.name == 'nt' else '> /dev/null 2>&1'
    subp = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    subp.wait()
    if DBG: print_log(subp)

    return subp

def gpcc_att_encode(filedir, bin_dir, posQuantscale=1, tmc3dir='tmc3_v23', cfgdir='cfgs/ford_01_q1mm', DBG=False):
    tmc3dir = os.path.join(rootdir, tmc3dir)
    cfgdir = os.path.join(rootdir, cfgdir)
    cmd = tmc3dir + ' --mode=0 ' \
        + ' --config='+cfgdir \
        + ' --positionQuantizationScale='+str(posQuantscale) \
        + ' --uncompressedDataPath='+filedir \
        + ' --compressedStreamPath='+bin_dir 
    # cmd += '> nul 2>&1' if os.name == 'nt' else '> /dev/null 2>&1'
    subp = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    subp.wait()
    if DBG: log_info=print_log(subp)

    return log_info,subp

def gpcc_att_decode(bin_dir, dec_dir,  tmc3dir='tmc3_v23', cfgdir='cfgs/ford_01_q1mm', DBG=False):
    tmc3dir = os.path.join(rootdir, tmc3dir)
    cfgdir = os.path.join(rootdir, cfgdir)
    cmd = tmc3dir + ' --mode=1 ' \
        + ' --config='+cfgdir \
        + ' --compressedStreamPath='+bin_dir \
        + ' --reconstructedDataPath='+dec_dir \
        + ' --outputBinaryPly=0'
    # cmd += '> nul 2>&1' if os.name == 'nt' else '> /dev/null 2>&1'
    subp = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    subp.wait()
    if DBG: log_info = print_log(subp)

    return log_info, subp

    
    

    
    

def print_log(p):
    tmp_str = ''
    c=p.stdout.readline()
    while c:
        print(c)
        tmp_str += c.decode('utf-8')
        c=p.stdout.readline()
        
    return tmp_str

if __name__ == '__main__':
    file_dir = '/home/old/huangwenjie/python_files/RAHT/exten_proj/my_exten/data_gen/att_plys_int/longdress_vox10_1052.ply'
    bin_dir = 'exp/00000001.bin'
    gpcc_att_encode(file_dir, bin_dir,DBG=True)
    
    dec_dir = 'exp/00000001.ply'
    gpcc_att_decode(bin_dir, dec_dir,DBG=True)
    
    