import os
import configparser
for i in [10,20,30,40,50,60,70,80,90]:
        config = configparser.ConfigParser()
        config.read('/home/ldd/project/SGL/conf/DisCom.ini')
        config.set('hyperparameters', 'user_com', str(i))

        with open('/home/ldd/project/SGL/conf/DisCom.ini', 'w') as configfile:
            config.write(configfile)
        config.read('/home/ldd/project/SGL/conf/Rec.ini')
        config.set('hyperparameters', 'user_com', str(i))

        with open('/home/ldd/project/SGL/conf/Rec.ini', 'w') as configfile:
            config.write(configfile)
        os.system('python main_dis.py')
        os.system('python main_rec.py')