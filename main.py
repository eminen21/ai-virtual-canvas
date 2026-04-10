import cv2
import mediapipe as mp
import numpy as np
import math

# 1. Başlangıç Ayarları ve Modern Renk Paleti
mp_hands = mp.solutions.hands
# Daha kararlı takip için hassasiyeti artırdık
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.9, min_tracking_confidence=0.9)
mp_draw = mp.solutions.drawing_utils

# Kamera Ayarları (HD Büyüklük)
cap = cv2.VideoCapture(0)
cap.set(3, 1280) # Genişlik
cap.set(4, 720)  # Yükseklik

# Tasarım Sabitleri
H, W = 720, 1280
PANEL_H = 120 # Kontrol panelinin yüksekliği
BUTON_R = 40  # Butonların yarıçapı
BUTON_Y = 60  # Butonların Y koordinatı

# Modern Renkler (BGR formatında)
RENKLER = [
    (230, 170, 0),   # Canlı Mavi
    (30, 200, 50),   # Güzel Yeşil
    (50, 50, 220),   # Tatlı Kırmızı
    (20, 220, 240),  # Parlak Sarı
    (240, 240, 240)  # Beyaz (Silgi)
]
RENK_ADLARI = ["MAVI", "YESIL", "KIRMIZI", "SARI", "SILGI"]

# Değişkenler
prev_x, prev_y = 0, 0
canvas = np.zeros((H, W, 3), np.uint8)
# Varsayılan renk: Mavi (Listeden seçtik)
secili_renk_index = 0 
firca_kalinligi = 10
silgi_kalinligi = 80

# Buton merkezlerini hesapla (Ekranı ortalayacak şekilde)
buton_merkezleri = []
toplam_buton = len(RENKLER)
ara_bosluk = (W - (toplam_buton * BUTON_R * 2)) // (toplam_buton + 1)
for i in range(toplam_buton):
    x = ara_bosluk + BUTON_R + i * (BUTON_R * 2 + ara_bosluk)
    buton_merkezleri.append((x, BUTON_Y))

# Şeffaf Arayüz Katmanı Oluştur
overlay = np.zeros((H, W, 3), np.uint8)

while True:
    success, img = cap.read()
    if not success: break
    img = cv2.flip(img, 1)
    
    # Arayüzü temizle (Her karede yeniden çizeceğiz)
    overlay = np.zeros((H, W, 3), np.uint8)
    
    # 2. Modern Arayüz Çizimi (Yuvarlak ve Şeffaf)
    # Panel Arka Planı (Hafif gri, şeffaf)
    cv2.rectangle(overlay, (0, 0), (W, PANEL_H), (50, 50, 50), cv2.FILLED)

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    parmak_x, parmak_y = 0, 0
    mod_secim = False

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lm = handLms.landmark
            # İşaret (8) ve Orta (12) parmak uçları
            x8, y8 = int(lm[8].x * W), int(lm[8].y * H)
            x12, y12 = int(lm[12].x * W), int(lm[12].y * H)
            parmak_x, parmak_y = x8, y8

            # Mod Kontrolü: İşaret ve Orta parmak havadaysa SEÇİM/GEZİNME
            if lm[8].y < lm[6].y and lm[12].y < lm[10].y:
                mod_secim = True
                prev_x, prev_y = 0, 0 # Çizmeyi bırak
                
                # İmleç Geri Bildirimi (Şeffaf halka)
                cv2.circle(overlay, (x8, y8), 15, RENKLER[secili_renk_index], 2)
                cv2.circle(overlay, (x12, y12), 15, RENKLER[secili_renk_index], 2)

            # Çizim Modu: Sadece İşaret parmağı havadaysa
            elif lm[8].y < lm[6].y:
                # Fırça ucu göstergesi (Dolu daire)
                cv2.circle(overlay, (x8, y8), firca_kalinligi // 2, RENKLER[secili_renk_index], cv2.FILLED)
                
                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = x8, y8

                # Silgi veya Renk
                guncel_renk = RENKLER[secili_renk_index]
                # Silgi ise siyah çiz (aslında tuvali temizler)
                if RENK_ADLARI[secili_renk_index] == "SILGI":
                    guncel_renk = (0, 0, 0)
                    guncel_kalinlik = silgi_kalinligi
                else:
                    guncel_kalinlik = firca_kalinligi
                    
                cv2.line(canvas, (prev_x, prev_y), (x8, y8), guncel_renk, guncel_kalinlik)
                prev_x, prev_y = x8, y8
            
            else:
                prev_x, prev_y = 0, 0

    # 3. Butonları Çiz ve Etkileşim Kontrolü
    for i, center in enumerate(buton_merkezleri):
        renk = RENKLER[i]
        
        # Hover (Üzerine Gelme) Efekti
        distance = math.sqrt((parmak_x - center[0])**2 + (parmak_y - center[1])**2)
        radius = BUTON_R + 10 if distance < BUTON_R else BUTON_R
        
        # Buton Dairesi
        cv2.circle(overlay, center, radius, renk, cv2.FILLED)
        
        # Seçili Buton Vurgusu (Beyaz halka)
        if i == secili_renk_index:
            cv2.circle(overlay, center, radius + 5, (255, 255, 255), 3)
            
        # Tıklama Kontrolü (Sadece seçim modunda)
        if mod_secim and distance < BUTON_R:
            secili_renk_index = i

    # 4. Görüntüleri Birleştirme (Şeffaflık Uygulama)
    # Arayüzü videonun üzerine şeffaf bir şekilde ekle
    img = cv2.addWeighted(img, 1.0, overlay, 0.4, 0)
    
    # Kanvastaki çizimleri ekle
    img_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, img_inv = cv2.threshold(img_gray, 10, 255, cv2.THRESH_BINARY_INV)
    img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, img_inv)
    img = cv2.bitwise_or(img, canvas)

    cv2.imshow("Modern AI Sanal Tuval 2.0", img)
    
    key = cv2.waitKey(1)
    # 'c' ile temizle, 'esc' ile çık
    if key & 0xFF == ord('c'):
        canvas = np.zeros((H, W, 3), np.uint8)
    elif key & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()
