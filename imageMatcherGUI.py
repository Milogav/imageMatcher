import numpy as np
import cv2
import tkinter as tk
import tkinter.filedialog
from PIL import Image,ImageTk
import os
from natsort import natsorted
from utils import *

def numpy2tkImg(img_array):
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    pil_img =  Image.fromarray(img_array)
    tk_img = ImageTk.PhotoImage(pil_img)
    return tk_img

class imageMatcher(object):

    def __init__(self,nFeatures = 300,initDir = '/home', title = 'ImageMatcher'):

        self.window = tk.Tk()
        self.window.title(title)
        self.window.rowconfigure(0, weight=1)
        self.window.columnconfigure(0, weight=1)
        self.frame = tk.Frame(self.window,background = 'white')
        self.frame.grid(row=0, column=0, sticky='NEWS')
        self.frame.bind('<Configure>',self.on_resize)
        
        self.initDir = initDir
        self.panels = dict()
        self.images = dict()
        self.databaseDir = None
        self.inputImgPath = None
        self.Matcher = None

        self.labelImg = tk.Label(self.frame,text='IMAGEN',background = 'white')
        self.labelImg.grid(row=0, column=0, sticky='NEWS',padx=70,pady=(10,0))
        
        self.labelMatchImg = tk.Label(self.frame,text='EMPAREJADA',background = 'white')
        self.labelMatchImg.grid(row=0, column=1, sticky='NEWS',padx=70,pady=(10,0))
        
        self._addCanvas(row = 1,column = 0)
        self._addCanvas(row = 1,column = 1)

        self.labelMatching = tk.Label(self.frame,text='MATCHING',background = 'white')
        self.labelMatching.grid(row=2, column=1, sticky='NEWS',padx=70,pady=(10,0))
        self._addCanvas(row = 3,column = 1,rowspan = 7,columnspan = 1)

        self.databaseButton = tk.Button(self.frame, text="Directorio de busqueda...", command=self.on_load_database, width=30)
        self.databaseButton.grid(row=4, column=0, sticky='NEW',padx=50,pady=(10,0))
        self.databasePathLabel = tk.Label(self.frame)
        self.databasePathLabel.grid(row=5, column=0, sticky='EW',padx=10,pady=(10,40))
        
        self.inputImgButton = tk.Button(self.frame, text="Imagen de entrada...", command=self.on_load_img, width=30)
        self.inputImgButton.grid(row=6, column=0, sticky='NEW',padx=50,pady=(10,0))
        self.inputImgPathLabel = tk.Label(self.frame)
        self.inputImgPathLabel.grid(row=7, column=0, sticky='EW',padx=10,pady=(10,40))

        self.runButton = tk.Button(self.frame, text="Ejecutar", command=self.on_run, width=30)
        self.runButton.grid(row=8, column=0, sticky='NEW',padx=50,pady=(10,0))
        
        self.outputLabel = tk.Label(self.frame,text="**Log del programa**\n\n",background = 'black',foreground='white')
        self.outputLabel.grid(row=9, column=0, sticky='NEWS',padx = 40,pady=50)

        self.clearButton = tk.Button(self.frame, text="Limpiar log", command=self.on_clear_log, width=30)
        self.clearButton.grid(row=10, column=0, sticky='NEW',padx=50,pady=10)

        self.imgVisualizerSpinbox = tk.Spinbox(self.frame,from_=0, to=0,command = self.on_spinbox_change,width = 10)
        self.imgVisualizerSpinbox.grid(row=10, column=1, sticky='NS',padx = 10,pady=10)

        self.featureDetector = cv2.ORB_create(nFeatures)
        self.Matcher = cv2.BFMatcher(cv2.NORM_L1,crossCheck=False)


        # # FLANN parameters
        # FLANN_INDEX_LSH = 6
        # index_params= dict(algorithm = FLANN_INDEX_LSH,
        #                         table_number = 12, # 12
        #                         key_size = 20,     # 20
        #                         multi_probe_level = 2) #2
        # search_params = dict(checks=50)   # or pass empty dictionary
        # self.Matcher = cv2.FlannBasedMatcher(index_params,search_params)

        # self.featureDetector = cv2.ORB_create(nFeatures)
        # self.Matcher = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=False)

    def _addCanvas(self,row,column,rowspan = 1,columnspan = 1,sticky = 'NEWS'):
        self.frame.rowconfigure(row, weight=1)
        self.frame.columnconfigure(column, weight=1)
        panel = tk.Canvas(self.frame, width = 400, height = 400,background = 'black')
        panel.grid(row = row,
                   column = column,
                   padx=10,
                   pady=10,
                   rowspan = rowspan,
                   columnspan = columnspan,
                   sticky=sticky)

        panelId = str(row)+str(column)
        self.panels[panelId] = panel
    
    def logPrint(self,text):
        self.outputLabel['text'] = text
        self.window.update()

    def on_resize(self,event):
        for tag in self.panels:
            panel = self.panels[tag]
            if tag in self.images.keys():
                img = self.images[tag]
                self.window.update()
                panelHeight = panel.winfo_height()
                panelWidth = panel.winfo_width()
                fitImg = self.imgFit(img,panelHeight,panelWidth)
                tkimg = numpy2tkImg(fitImg)
                panel.create_image(0, 0, image=tkimg, anchor='nw')
                panel.image = tkimg
    
    def on_clear_log(self):
        self.outputLabel['text'] = '**Log del programa**\n\n'
        self.window.update()

    def on_load_database(self):
        dirname = tk.filedialog.askdirectory(title='Directorio de busqueda...',initialdir=self.initDir)
        ### Se debe introducir un directorio que contenga carpetas. 
        ### Cada una de estas capetas del directorio debe contener las imagenes para cada clase de la base de datos.
        if dirname and not os.path.isdir(dirname):
            self.logPrint('ERROR, ruta a directorio de busqueda no valido\n')
        else:
            self.databaseDir = dirname
            self.databasePathLabel['text'] = dirname

            folders = filelist(self.databaseDir)
            self.trainImgPaths = list()
            self.trainImgDescriptors = list()
            self.trainDescriptorsIdx = list()
            self.trainDescriptorsClass = list()
            cont = 0
            l = str(len(folders))
            for c,classFolder in enumerate(folders): 
                imgFiles = filelist(classFolder)
                for imgPath in imgFiles:
                    self.logPrint('Cargando directorio de busqueda...\nCarpeta: '+str(c)+'/'+l+'  -  Nº Imagen: '+str(cont)+' ...\n')

                    img = cv2.imread(imgPath)
                    _,des = self.featureDetector.detectAndCompute(img,None)
                    if des is not None:
                        self.trainImgPaths.append(imgPath)
                        self.trainImgDescriptors.append(des)
                        self.trainDescriptorsIdx += [cont]*len(des)
                        cont += 1
                        self.trainDescriptorsClass += [c]*len(des)
                    
            
            self.trainImgDescriptors = np.vstack(self.trainImgDescriptors)
            self.trainDescriptorsIdx = np.array(self.trainDescriptorsIdx)
            self.trainDescriptorsClass = np.array(self.trainDescriptorsClass)
    
            n = len(self.trainImgPaths)
            self.imgVisualizerSpinbox.config(to=n-1)
            
            self.initDir = dirname
            img = cv2.imread( self.trainImgPaths[0])
            self.assignImg(img,'31')
            self.logPrint('...Hecho\n')

    def on_load_img(self):
        filename = tk.filedialog.askopenfilename(title='Imagen de entrada...',initialdir=self.initDir)
        if filename and not os.path.isfile(filename):
            self.logPrint('ERROR, ruta imagen a emparejar no valida\n')
        else:
            self.inputImgPath = filename
            self.inputImgPathLabel['text'] = filename
            img = cv2.imread(filename)
            self.assignImg(img,'10')
            self.initDir = os.path.dirname(self.inputImgPath)
    
    def on_spinbox_change(self):
        if self.databaseDir:
            val = int(self.imgVisualizerSpinbox.get())
            img = cv2.imread( self.trainImgPaths[val])
            self.assignImg(img,'31')
   
    def match(self,nBest = None):

        queryImg = self.images['10']
        _, des = self.featureDetector.detectAndCompute(queryImg,None)
        matches = self.Matcher.knnMatch(des,self.trainImgDescriptors, k=2)

        #### filter good matches #####
        goodMatches = list()
        for m in matches:
             try:
                 if m[0].distance < 0.75*m[1].distance:
                     goodMatches.append([m[0].queryIdx,m[0].trainIdx,m[0].distance,-1,-1]) 
             except:
                 pass
        goodMatches = np.array(goodMatches,int)
        ordPos = np.argsort(goodMatches[:,2])
        goodMatches = goodMatches[ordPos,:]
        matchImgClasses = self.trainDescriptorsClass[ goodMatches[:,1] ]
        goodMatches[:,3] = matchImgClasses
        matchImgIdx = self.trainDescriptorsIdx[ goodMatches[:,1] ]
        goodMatches[:,4] = matchImgIdx
        r = goodMatches.shape[0]
        if nBest is not None and nBest < r:
            goodMatches = goodMatches[0:nBest,:]
        return goodMatches
    
    def on_run(self):
        if os.path.isfile(self.inputImgPath) and os.path.isdir(self.databaseDir):
            self.logPrint('Cotejando imagen con la base de datos introducida...\n')
            queryImg = self.images['10']

            matches = self.match()
            matchIdx = matches[:,4]
            bestId = topN(matchIdx)[0]
            matchImgPath = self.trainImgPaths[bestId]
            matchImg = cv2.imread(matchImgPath)
            self.assignImg(matchImg,'11')

            ##### draw descriptor matching
            kp1, des1 = self.featureDetector.detectAndCompute(queryImg,None)
            kp2, des2 = self.featureDetector.detectAndCompute(matchImg,None)
            matches = self.Matcher.match(des1,des2)
            draw_params = dict(matchColor = (0,255,0),
                              singlePointColor = (255,0,0),
                              flags = 0)
            display = cv2.drawMatches(queryImg,kp1,matchImg,kp2,matches,None,**draw_params)
            self.assignImg(display,'31')
            self.logPrint('...Hecho\n')

    def imgFit(self,img,height,width):
        h,w = img.shape[0:2]
        aspectRatio = h/w
        if h >= w:
            nW = int(height / aspectRatio)
            if nW > width:
                nH = int(aspectRatio * width)
                nW = width
            else:
                nH = height   
        else:
            nH = int(aspectRatio * width)
            if nH > height:
                nW = int(height / aspectRatio)
                nH = height
            else:
                nW = width
        img = cv2.resize(img,(nW,nH),interpolation = cv2.INTER_LANCZOS4)
        return img

    def assignImg(self,img,panelId,resize=True):
        panel = self.panels[panelId]
        self.images[panelId] = img
        self.window.update()
        panelWidth = panel.winfo_width()
        panelHeight = panel.winfo_height()
        fitImg = self.imgFit(img,panelHeight,panelWidth)
        tkimg = numpy2tkImg(fitImg)
        panel.create_image(0, 0, image=tkimg, anchor='nw')
        panel.image = tkimg
         

scriptPath = os.path.dirname(os.path.realpath(__file__))
IM = imageMatcher(nFeatures=500,initDir = scriptPath)

# Run the window loop
IM.window.mainloop()
