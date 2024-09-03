import numpy as np
import cv2 as cv

blurry_moon = cv.imread('blurry_moon.tif')
blurry_moon = cv.cvtColor(blurry_moon, cv.COLOR_BGR2GRAY)

camera_man = cv.imread('cameraman.tif')
camera_man = cv.cvtColor(camera_man, cv.COLOR_BGR2GRAY)

lenna = cv.imread('Lenna.png')
lenna = cv.cvtColor(lenna, cv.COLOR_BGR2GRAY)

def convolucao(img, mask):
    # Inverter a máscara
    mask = np.flip(mask)
    
    # Tamanho da imagem
    img_x, img_y = img.shape
    
    # Tamanho da máscara
    mask_x, mask_y = mask.shape
    
    # Tamanho da borda
    borda_x = mask_x // 2
    borda_y = mask_y // 2

    # Imagem de saída
    img_saida = np.pad(img, ((borda_x, borda_x), (borda_y, borda_y)), mode='empty')

    # Vetores de índices das janelas
    i = np.arange(img_x)
    j = np.arange(img_y)
    k = np.arange(mask_x)
    l = np.arange(mask_y)

    # Cria a matriz de janelas deslizantes
    cortes = img_saida[i[:, None, None, None] + k[None, None, :, None],
                       j[None, :, None, None] + l[None, None, None, :]]
    
    # Convolução
    return np.sum(cortes * mask[None, None, :, :], axis=(2, 3))

def clip(img):
    return np.clip(img, 0, 255)

def filtro_media_simples(img, tam):
    mask = np.full((tam,tam), 1/(tam*tam))
    return convolucao(img, mask)

def filtro_high_boost(img, k):
    img_borrada = filtro_media_simples(img, 3)
    mascara = img - img_borrada
    return clip(img + k * mascara)

cv.imshow('blurry_moon original', blurry_moon)
cv.imshow('blurry_moon processada', filtro_high_boost(blurry_moon, 5).astype(np.uint8))

cv.imshow('cameraman original', camera_man)
cv.imshow('cameraman processada', filtro_high_boost(camera_man, 5).astype(np.uint8))

cv.imshow('lenna original', lenna)
cv.imshow('lenna processada', filtro_high_boost(lenna, 5).astype(np.uint8))

cv.waitKey(0)
cv.destroyAllWindows()