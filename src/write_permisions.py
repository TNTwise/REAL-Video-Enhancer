import os
import src.thisdir
homedir =  os.path.expanduser(r"~")
thisdir = src.thisdir.thisdir()
def check_for_write_permissions(dir):

        if 'FLATPAK_ID' in os.environ:
            import subprocess

            command = f'{thisdir}/bin/flatpak info --show-permissions io.github.tntwise.REAL-Video-Enhancer'

            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            print(result.stderr)
            output = result.stdout.split('\n')
            output_2=[]
            for i in output:
                if len(i) > 0 and i != '\n':
                    output_2.append(i)
            print(output_2)
            directories_with_permissions=[]
            for i in output_2:
                print(i)
                if 'filesystems=' in i:
                    i=i.split(';')
                    print(i)
                    s=[]
                    for e in  i:
                        if len(e) > 0 and i != '\n':
                            s.append(e)
                    for j in s:
                        j=j.replace('filesystems=','')
                        j=j.replace('xdg-',f'{homedir}')
                        j+='/'
                        directories_with_permissions.append(j)
                    break
            for i in directories_with_permissions:
                if dir.lower() in i.lower() or 'io.github.tntwise.real-video-enhancer' in dir.lower():
                    print(f'I: {i}')
                    print(f'Dir: {dir}')
                    return True
                else:
                    print(f'Dir: {dir}')
                    print(f'I: {i}')
                    return False
        else:
                if os.access(dir, os.R_OK) and os.access(dir, os.W_OK):
                    print('has access')
                    return True
                return False