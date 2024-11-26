import glob
import os

if __name__ == "__main__":
    protocols = [os.path.basename(x).split('.')[0] for x in glob.glob('rhfe_gromacs/input/mdpath/*.X.mdp')]
    print(protocols)
    for protocol in protocols:
        template = f'rhfe_gromacs/input/mdpath/{protocol}.X.mdp'

        with open(template, 'r') as f:
            content = f.read()

        for i in range(20):
            with open(f'rhfe_gromacs/input/mdpath/files/{protocol}.{i}.mdp', 'w') as f:
                if i <= 14:
                    content_new = content.replace('XXX', 'AB')
                    vdw = '0.00 0.00 0.00 0.00 0.00 0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.90 1.00'
                    coul = '0.00 0.25 0.50 0.75 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00'
                    content_new = content_new.replace('VVV', vdw)
                    content_new = content_new.replace('WWW', coul)
                    content_new = content_new.replace('YYY', str(i))
                elif i >= 15:
                    content_new = content.replace('XXX', 'B')
                    vdw = '0.00 0.00 0.00 0.00 0.00'
                    coul = '0.00 0.25 0.50 0.75 1.00'
                    content_new = content_new.replace('VVV', vdw)
                    content_new = content_new.replace('WWW', coul)
                    content_new = content_new.replace('YYY', str(i-15))
                if i ==0 or i==19:
                    content_new = content_new.replace('ZZZ', '500')
                else:
                    content_new = content_new.replace('ZZZ', '5000')

                f.write(content_new)