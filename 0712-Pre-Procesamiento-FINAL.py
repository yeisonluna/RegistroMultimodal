# -*- coding: utf-8 -*-
"""
PRE-PROCESAMIENTO DE LAS IMÁGENES DEL TRABAJO DE GRADO "Sistema de registro 
multimodal de imágenes de presión plantar y escáner digital 
para la caracterización del pie diabético en el laboratorio BASPI/FootLab.

                    Yeison Estiven Luna Zuluaga
                Pontificia Universidad Javeriana
                     Facultad de Ingeniería
                Carrera de Ingeniería Electrónica


El Pre-procesamiento se hace por cada voluntario, y cada una de las muestras 
del mismo

Entrada: Imagen de presión plantar (una imagen), Imágenes de podoscopio de cada 
pie (dos imágenes)

Salida: Imágenes de presión plantar pre procesadas por cada pie (dos imágenes),
        Imágenes de podoscopio por cada pie(dos imágenes)


"""

#*********************** Importación de bibliotecas **************************#
import cv2                                                                    #
import numpy as np                                                            #
import matplotlib.pyplot as plt                                               #
#*****************************************************************************#

#-------------- Inicio Bloque de Pre-Procesamiento Individual - Presión Plantar -------------#


#***************************** Función SeparacionPies ************************#
def SeparacionPies(ImgPresion):        
    
    # Pie izquierdo    
    ImgPresion_izq = ImgPresion[:,0:228,:]
    
    #Pie derecho 
    ImgPresion_der = ImgPresion[:,227:455,:]   
    
    return ImgPresion_izq, ImgPresion_der
#***********************Fin Función SeparacionPies*****************************#
    
    
#************************** Función ReplicarBordesPresion ********************#
def ReplicarBordesPresion(img_izq,img_der):    
    
    # Pie izquierdo
    img_izq_bordes = cv2.copyMakeBorder(img_izq,0,0,113,114,cv2.BORDER_REPLICATE)
    
    # Pie derecho
    img_der_bordes = cv2.copyMakeBorder(img_der,0,0,113,114,cv2.BORDER_REPLICATE)   
    
    return img_izq_bordes,img_der_bordes
#************************** Fin Función ReplicarBordesPresion ****************#
    

    
img1 = cv2.imread("C:/Users/yeiso/Desktop/Universidad/Trabajo de grado/Prueba 01-06-2021/Paciente_1010244758_Presion-Promedio_2a.TIFF") 



#*********** Resultados primer bloque individual presión *********************#
img1_izq, img1_der = SeparacionPies(img1)
img1_izq_bordes, img1_der_bordes = ReplicarBordesPresion(img1_izq,img1_der)


cv2.imshow("imagen presion",img1)
cv2.imshow("Imagen izq",img1_izq)
cv2.imshow("Imagen der",img1_der)


cv2.imshow("Imagen izq bordes",img1_izq_bordes)
cv2.imshow("Imagen der bordes",img1_der_bordes)
#*****************************************************************************#




#----------------- Fin Bloque de Pre-Procesamiento Individual - Presión Plantar --------------#




#------------- Inicio Bloque de Pre-Procesamiento Individual - Podoscopio --------------------#

#****************** Inicio Transformacion de perspectiva *********************#
def clics(event,x,y,flags,param):
	global puntos
	if event == cv2.EVENT_LBUTTONDOWN:
		puntos.append([x,y])

#***************** Pie izq ****************
puntos = []
imagen = cv2.imread("C:/Users/yeiso/Desktop/Universidad/Trabajo de grado/Prueba 01-06-2021/IMG_6848.JPG")
scale_percent = 15 # percent of original size
width = int(imagen.shape[1] * scale_percent / 100)
height = int(imagen.shape[0] * scale_percent / 100)
dim = (width, height)
 
# resize image
imagen = cv2.resize(imagen, dim, interpolation = cv2.INTER_AREA)

aux = imagen.copy()
cv2.namedWindow('Imagen')
cv2.setMouseCallback('Imagen',clics)

while True:

	if len(puntos) == 4:
		#uniendo4puntos(puntos)
		pts1 = np.float32([puntos])
		pts2 = np.float32([[0,0], [455,0], [0,419], [455,419]])

		M = cv2.getPerspectiveTransform(pts1,pts2)
		dst = cv2.warpPerspective(imagen, M, (455,419))

		cv2.imshow('dst', dst)
	cv2.imshow('Imagen',imagen)
	
	k = cv2.waitKey(1) & 0xFF
	if k == ord('n'):
		imagen = aux.copy()
		puntos = []
		
	elif k == 27:
		break


#******************** Pie derecho ***************************#

puntos = []
imagen = cv2.imread("C:/Users/yeiso/Desktop/Universidad/Trabajo de grado/Prueba 01-06-2021/IMG_6848_der.JPG")
scale_percent = 15 # percent of original size
width = int(imagen.shape[1] * scale_percent / 100)
height = int(imagen.shape[0] * scale_percent / 100)
dim = (width, height)
 
# resize image
imagen = cv2.resize(imagen, dim, interpolation = cv2.INTER_AREA)

aux = imagen.copy()
cv2.namedWindow('Imagen')
cv2.setMouseCallback('Imagen',clics)

while True:

	if len(puntos) == 4:
		#uniendo4puntos(puntos)
		pts1 = np.float32([puntos])
		pts2 = np.float32([[0,0], [455,0], [0,419], [455,419]])

		M = cv2.getPerspectiveTransform(pts1,pts2)
		dst2 = cv2.warpPerspective(imagen, M, (455,419))

		cv2.imshow('dst2', dst2)
	cv2.imshow('Imagen',imagen)
	
	k = cv2.waitKey(1) & 0xFF
	if k == ord('n'):
		imagen = aux.copy()
		puntos = []
		
	elif k == 27:
		break
#******************** Fin Transformacion de perspectiva **********************#
        



#******************** Inicio Replicar Bordes Podoscopio **********************#
def ReplicarBordesPodoscopio(izq,der):    
    
    # Pie izquierdo
    izq_bordes = cv2.copyMakeBorder(izq,18,18,0,0,cv2.BORDER_REPLICATE)
    
    # Pie derecho
    der_bordes = cv2.copyMakeBorder(der,18,18,0,0,cv2.BORDER_REPLICATE)   
    
    return izq_bordes,der_bordes
#******************** Fin Replicar Bordes Podoscopio *************************#

#*********** Resultados bloque individual podoscopio *************************#
img2_perspectiva = dst
img3_perspectiva = dst2


cv2.imshow("Imagen podoscopio izq perspectiva",img2_perspectiva)
cv2.imshow("Imagen podoscopio der perspectiva",img3_perspectiva)


img2_izq_bordes,img3_der_bordes = ReplicarBordesPodoscopio(img2_perspectiva,img3_perspectiva)

cv2.imshow("Imagen podoscopio izq bordes",img2_izq_bordes)
cv2.imshow("Imagen podoscopio der bordes",img3_der_bordes)




#******************************************************************************

#--------------- Fin Bloque de Pre-Procesamiento Individual - Podoscopio ---------------------#



#--------- Inicio Pre procesamiento Conjunto - Estimación de variables s,x,y,theta -----------#

def EncontrarRectanguloPodoscopio(imagen):
    """
    Parameters
    ----------
    imagen : Imagen de entrada
        

    Returns
    ---
    rectangulo : Tupla con centro de rectángulo (x,y), ancho y largo (w,h) y angulo de rotacion
    -------
    gris : Imagen de salida con el rectangulo dibujado

    """
    gris = cv2.cvtColor(imagen,cv2.COLOR_BGR2GRAY)    
    _, th = cv2.threshold(gris,150,255,cv2.THRESH_BINARY) 
    
    
    kernel = np.ones((9,9),np.uint8)
    kernel2 = np.ones((6,6),np.uint8)
    kernel3 = np.ones((8,8),np.uint8)
                      
    apertura = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
    dilatar = cv2.morphologyEx(apertura,cv2.MORPH_DILATE,kernel2)
    dilatar2 = cv2.morphologyEx(dilatar,cv2.MORPH_DILATE,kernel)
    erosion = cv2.morphologyEx(dilatar2,cv2.MORPH_ERODE,kernel3)
    
    #contornos,jerarquia = cv2.findContours(apertura,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    contornos,jerarquia = cv2.findContours(erosion,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    contornos = sorted(contornos, key=cv2.contourArea, reverse=True)[:3]
    #cv2.drawContours(gris,contornos,-1,(0,255,255),2)
    
    
    
    for c in contornos:
        area=cv2.contourArea(c)
        if area > 2000:
            x, y, w, h = cv2.boundingRect(c)
            #cv2.rectangle(gris, (x, y), (x+w, y+h), (128, 0, 0), 2)
            rectangulo = cv2.minAreaRect(c)
            box = np.int0(cv2.boxPoints(rectangulo))
            cv2.drawContours(gris, [box], 0, (255,255,20), 2) # OR
            
    rect = rectangulo
    return rect, gris



def EncontrarRectanguloPresion(imagen):
    """
    Parameters
    ----------
    imagen : Imagen de entrada
        

    Returns
    ---
    rectangulo : Tupla con centro de rectángulo (x,y), ancho y largo (w,h) y angulo de rotacion
    -------
    gris : Imagen de salida con el rectangulo dibujado

    """
    gris = cv2.cvtColor(imagen,cv2.COLOR_BGR2GRAY)    
    _, th = cv2.threshold(gris,25,255,cv2.THRESH_BINARY) 
    
    
    kernel = np.ones((9,9),np.uint8)
    kernel2 = np.ones((6,6),np.uint8)
    kernel3 = np.ones((8,8),np.uint8)

    apertura = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
    dilatar = cv2.morphologyEx(apertura,cv2.MORPH_DILATE,kernel2)
    dilatar2 = cv2.morphologyEx(dilatar,cv2.MORPH_DILATE,kernel)
    erosion = cv2.morphologyEx(dilatar2,cv2.MORPH_ERODE,kernel3)
    
    #contornos,jerarquia = cv2.findContours(apertura,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    contornos,jerarquia = cv2.findContours(erosion,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    contornos = sorted(contornos, key=cv2.contourArea, reverse=True)[:3]
    #cv2.drawContours(gris,contornos,-1,(0,255,255),2)
    
    
    
    for c in contornos:
        area=cv2.contourArea(c)
        if area > 1500:
            x, y, w, h = cv2.boundingRect(c)
            #cv2.rectangle(gris, (x, y), (x+w, y+h), (128, 0, 0), 2)
            rectangulo = cv2.minAreaRect(c)
            box = np.int0(cv2.boxPoints(rectangulo))
            cv2.drawContours(gris, [box], 0, (255,255,20), 2) # OR
            
    rect = rectangulo
    return rect, gris




    



# Obteniendo rectangulos de las imagenes de presión
rect_presion_izq, img_presion_izq_rect = EncontrarRectanguloPresion(img1_izq_bordes)
rect_presion_der, img_presion_der_rect = EncontrarRectanguloPresion(img1_der_bordes)

# Obteniendo rectangulos de las imagenes de podoscopio
rect_podos_izq, img_podos_izq_rect = EncontrarRectanguloPodoscopio(img2_izq_bordes)
rect_podos_der, img_podos_der_rect = EncontrarRectanguloPodoscopio(img3_der_bordes)


cv2.imshow("Imagen presion pie izquierdo rectangulo",img_presion_izq_rect)
cv2.imshow("Imagen presion pie derecho rectangulo",img_presion_der_rect)

cv2.imshow("Imagen podoscopio pie izquierdo rectangulo",img_podos_izq_rect)
cv2.imshow("Imagen podoscopio pie derecho rectangulo",img_podos_der_rect)



# Estimación de escala 

# x_presion_izq = rect_presion_izq[0][0]
# y_presion_izq = rect_presion_izq[0][1]
# w_presion_izq = rect_presion_izq[1][0]
# h_presion_izq = rect_presion_izq[1][1]
# angulo_presion_izq = rect_presion_izq[2]

# x_podos_izq = rect_podos_izq[0][0]
# y_podos_izq = rect_podos_izq[0][1]
# w_podos_izq = rect_podos_izq[1][0]
# h_podos_izq = rect_podos_izq[1][1]
# angulo_podos_izq = rect_podos_izq[2]
#factorEscala = h_podos_izq/h_presion_izq

factorEscala = (rect_podos_izq[1][1])/(rect_presion_izq[1][1])


#Estimación de traslacion x pie izq y der
factorTraslacion_x_izq = (rect_podos_izq[0][0]-rect_presion_izq[0][0])
factorTraslacion_x_der = (rect_podos_der[0][0]-rect_presion_der[0][0])

#Estimación de traslacion y pie izq y der
factorTraslacion_y_izq = (rect_podos_izq[0][1]-rect_presion_izq[0][1])
factorTraslacion_y_der = (rect_podos_der[0][1]-rect_presion_der[0][1])


#Estimacion angulo pie izq y der
factorAngulo_izq = rect_podos_izq[2]-rect_presion_izq[2]
factorAngulo_der = rect_podos_der[2]-rect_presion_der[2]


#Transformacion de imagenes de presion con valores estimados de variables



rows, cols = 455, 455

# Transformacion de escala para imagen de presión
M_escala = cv2.getRotationMatrix2D((cols/2,rows/2),0,factorEscala)
imagen_escala_presion_izq = cv2.warpAffine(img1_izq_bordes, M_escala, (cols,rows))
imagen_escala_presion_der = cv2.warpAffine(img1_der_bordes, M_escala, (cols,rows))


# Transformación de traslacion para pie izquierdo y derecho
M_traslacion_izq = np.float32([[1,0,factorTraslacion_x_izq],[0,1,factorTraslacion_y_izq]])
M_traslacion_der = np.float32([[1,0,factorTraslacion_x_der],[0,1,factorTraslacion_y_der]])

imagen_traslacion_presion_izq = cv2.warpAffine(imagen_escala_presion_izq, M_traslacion_izq, (cols,rows))
imagen_traslacion_presion_der = cv2.warpAffine(imagen_escala_presion_der, M_traslacion_der, (cols,rows))

# Transformacion de rotacion para pie izq y der
M_rotacion_izq = cv2.getRotationMatrix2D((cols/2,rows/2),factorAngulo_izq,1)
M_rotacion_der = cv2.getRotationMatrix2D((cols/2,rows/2),factorAngulo_der,1)

imagen_final_presion_izq = cv2.warpAffine(imagen_traslacion_presion_izq,M_rotacion_izq,(cols,rows))
imagen_final_presion_der = cv2.warpAffine(imagen_traslacion_presion_der,M_rotacion_der,(cols,rows))


# Imagenes pre-procesadas total de presion
cv2.imshow("Imagen final presion pie izquierdo",imagen_final_presion_izq)
cv2.imshow("Imagen final presion pie derecho",imagen_final_presion_der)


cv2.imwrite("C:/Users/yeiso/Desktop/Universidad/Trabajo de grado/Prueba 01-06-2021/Pruebas finales/PodoscopioPreProcesadaFinalIzq.JPG",img2_izq_bordes)
cv2.imwrite("C:/Users/yeiso/Desktop/Universidad/Trabajo de grado/Prueba 01-06-2021/Pruebas finales/PodoscopioPreProcesadaFinalDer.JPG",img3_der_bordes)
cv2.imwrite("C:/Users/yeiso/Desktop/Universidad/Trabajo de grado/Prueba 01-06-2021/Pruebas finales/PresionPreProcesadaFinalIzq.JPG",imagen_final_presion_izq)
cv2.imwrite("C:/Users/yeiso/Desktop/Universidad/Trabajo de grado/Prueba 01-06-2021/Pruebas finales/PresionPreProcesadaFinalDer.JPG",imagen_final_presion_der)


#------------ Fin Pre procesamiento Conjunto - Estimación de variables s,x,y,theta -----------#




cv2.waitKey(0)
cv2.destroyAllWindows()