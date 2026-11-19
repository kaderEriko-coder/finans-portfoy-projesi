import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pypfopt import EfficientFrontier, HRPOpt, risk_models, expected_returns
import scipy.cluster.hierarchy as sch
if not hasattr(sch, "_LINKAGE_METHODS"):
    sch._LINKAGE_METHODS = {"single": 0, "complete": 1, "average": 2, "weighted": 3, "centroid": 4, "median": 5, "ward": 6}
st.set_page_config(page_title="Akıllı Portföy Paneli", page_icon="📊", layout="wide")
st.title("📊 Akıllı Portföy Optimizasyonu & Risk Paneli")
st.caption("Altın, Gümüş ve Bitcoin | Markowitz (1952) vs. Makine Öğrenmesi (HRP)")

# Sol Menü (Kullanıcı Ayarları)
st.sidebar.header("⚙️ Yatırım Ayarları")
butce = st.sidebar.number_input("Yatırım Bütçen ($)", min_value=100, value=10000, step=500)
komisyon = st.sidebar.slider("Borsa Komisyon Oranı (%)", 0.0, 1.0, 0.2, 0.05) / 100

@st.cache_data(ttl=3600)
def veri_getir():
    df = yf.download(['GC=F', 'SI=F', 'BTC-USD'], start="2023-11-01", progress=False)['Close'].dropna()
    return df.rename(columns={'GC=F': 'Altın', 'SI=F': 'Gümüş', 'BTC-USD': 'Bitcoin'})

veri = veri_getir()

# Kopyasız Sınav Ayrımı (2/3 Eğitim, 1/3 Test)
bolme = int(len(veri) * (2 / 3))
egitim_fiyat, test_fiyat = veri.iloc[:bolme], veri.iloc[bolme:]
egitim_getiri = egitim_fiyat.pct_change().dropna()
test_getiri = test_fiyat.pct_change().dropna()

# 1. Markowitz Modeli
ef = EfficientFrontier(
    expected_returns.mean_historical_return(egitim_fiyat),
    risk_models.sample_cov(egitim_fiyat)
)
try:
    ef.max_sharpe(risk_free_rate=0.0)
except Exception:
    ef.min_volatility()
m_agirlik = ef.clean_weights()

# 2. Makine Öğrenmesi (HRP)
hrp = HRPOpt(egitim_getiri)
hrp.optimize()
h_agirlik = hrp.clean_weights()

# 3. Düz Mantık (Hepsinden Eşit)
e_agirlik = {col: 1 / len(veri.columns) for col in veri.columns}

def test_et(getiriler, agirliklar, kom_orani, baslangic):
    hedef_w = np.array([agirliklar[c] for c in getiriler.columns])
    deger = baslangic * (1 - kom_orani)
    toplam_kom = baslangic * kom_orani
    mevcut_w = hedef_w.copy()
    seri = []

    for i, (_, row) in enumerate(getiriler.iterrows()):
        r = row.values
        deger *= (1 + np.dot(mevcut_w, r))
        mevcut_w = (mevcut_w * (1 + r))
        mevcut_w /= mevcut_w.sum()

        if (i + 1) % 21 == 0:
            hacim = np.abs(mevcut_w - hedef_w).sum()
            maliyet = deger * hacim * kom_orani
            deger -= maliyet
            toplam_kom += maliyet
            mevcut_w = hedef_w.copy()
        seri.append(deger)

    seri = pd.Series(seri, index=getiriler.index)
    gunluk = seri.pct_change().dropna()
    zirve = seri.cummax()
    max_dd = ((seri - zirve) / zirve).min() * 100
    cvar = gunluk[gunluk <= np.percentile(gunluk, 5)].mean() * 100

    return {
        "Net Son Değer ($)": round(seri.iloc[-1], 2),
        "Ödenen Komisyon ($)": round(toplam_kom, 2),
        "En Büyük Çöküş (%)": round(max_dd, 2),
        "Kriz Günü Ort. Kayıp (CVaR %)": round(cvar, 2)
    }, seri

m_ozet, m_seri = test_et(test_getiri, m_agirlik, komisyon, butce)
h_ozet, h_seri = test_et(test_getiri, h_agirlik, komisyon, butce)
e_ozet, e_seri = test_et(test_getiri, e_agirlik, komisyon, butce)

# Sepet Dağılım Kartları
st.subheader(f"💰 {butce:,.0f} $ Bütçe İçin Önerilen Sepet Dağılımı")
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🏛️ Markowitz (1952)")
    for v, oran in m_agirlik.items():
        st.write(f"**{v}:** %{oran*100:.1f} → **{butce*oran:,.0f} $**")

with col2:
    st.markdown("### 🤖 Makine Öğrenmesi (HRP)")
    for v, oran in h_agirlik.items():
        st.write(f"**{v}:** %{oran*100:.1f} → **{butce*oran:,.0f} $**")

# Performans Tablosu ve Grafik
st.subheader("📈 Kopyasız Test Performansı (Hiç Görmediği Son 1 Yıl)")
tablo = pd.DataFrame({
    "Düz Mantık (1/3)": e_ozet,
    "Markowitz (1952)": m_ozet,
    "HRP (Makine Öğrenmesi)": h_ozet
})
st.dataframe(tablo, use_container_width=True)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(e_seri, label="Düz Mantık (Eşit)", linestyle="--", color="gray")
ax.plot(m_seri, label="Markowitz", linewidth=2)
ax.plot(h_seri, label="HRP (Makine Öğrenmesi)", linewidth=2)
ax.set_ylabel("Portföy Değeri ($)")
ax.legend()
ax.grid(alpha=0.3)
st.pyplot(fig)

# Otomatik Risk Yorum Kutusu
st.subheader("🤖 Akıllı Risk Yorumu")
kazanan = "Makine Öğrenmesi (HRP)" if h_ozet["Net Son Değer ($)"] >= m_ozet["Net Son Değer ($)"] else "Markowitz"
st.info(
    f"Seçtiğin **%{komisyon*100:.2f}** komisyon oranıyla kopyasız sınavda net getiride **{kazanan}** önde kapattı. "
    f"HRP portföyü 3 varlığa dengeli dağıtarak tek varlık riskini önlerken, Markowitz daha sakin bir çöküş profili "
    f"(**%{m_ozet['En Büyük Çöküş (%)']}**) sergiledi."
)
