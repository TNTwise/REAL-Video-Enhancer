import os
import src.thisdir
homedir =  os.path.expanduser(r"~")
thisdir = src.thisdir.thisdir()
def check_for_write_permissions(dir):
        i=0
        if 'FLATPAK_ID' in os.environ or i==1:
            import subprocess

            command = f'cat /.flatpak-info'
            #command = 'flatpak info --show-permissions io.github.tntwise.REAL-Video-Enhancer'
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            output = result.stdout.split('\n')
            output_2=[]
            for i in output:
                if len(i) > 0 and i != '\n':
                    output_2.append(i)
            directories_with_permissions=[]
            for i in output_2:
                if 'filesystems=' in i:
                    i=i.split(';')
                    s=[]
                    for e in  i:
                        if len(e) > 0 and i != '\n':
                            s.append(e)
                    for j in s:
                        j=j.replace('filesystems=','')
                        if 'xdg-download' == j:
                            j=f'{homedir}/Downloads'
                        j=j.replace('xdg-',f'{homedir}/')
                        
                        directories_with_permissions.append(j)
            for i in directories_with_permissions:
                if i.lower() in dir.lower() or 'io.github.tntwise.real-video-enhancer' in dir.lower():
                    return True
                else:
                    if '/run/user/1000/doc/' in dir:
                        i=i.replace('/run/user/1000/doc/','')
                        i=i.split('/')
                        permissions_dir=''
                        for index in range(len(i)):
                            if index != 0:
                                permissions_dir+=f'{i[index]}/'
                            
                        if homedir not in permissions_dir:
                            i=f'{homedir}/{permissions_dir}'
                        else:
                            i=permissions_dir
                        print(i)
                    if i.lower() in dir.lower() or 'io.github.tntwise.real-video-enhancer' in dir.lower():
                        return True
                    return False
        else:
                if os.access(dir, os.R_OK) and os.access(dir, os.W_OK):
                    print('has access')
                    return True
                return False