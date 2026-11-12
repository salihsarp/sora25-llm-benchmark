"""Prompt conditions.

closed_book : task instructions only; the model must rely on what it knows of SORA 2.5.
reference   : the same instructions plus a reference card restating the SORA 2.5 rules and table values
              (our own structured wording; numbers are facts from the JARUS tables).
extract     : the model only extracts operation parameters into a fixed schema; the deterministic
              engine (sora25) computes the result ("LLM + calculator" design).
"""

ANSWER_KEYS = ("igrc", "final_grc", "initial_arc", "residual_arc", "tmpr", "sail")

SYSTEM_EN = """You are an aviation safety specialist. Perform a JARUS SORA 2.5 assessment (Specific Operations Risk Assessment, edition 2.5, JARUS 2024) of the UAS operation described by the user. Determine: the intrinsic ground risk class (iGRC, Step 2), the final GRC (Step 3), the initial air risk class (ARC, Step 4), the residual ARC (Step 5), the tactical mitigation performance requirement (TMPR, Step 6) and the SAIL (Step 7).

Rules for this task:
- Credit only the ground-risk mitigations that the operator explicitly claims. Treat the stated robustness (or integrity and assurance levels) of each claimed mitigation as already substantiated and accepted by the authority; do not re-assess the evidence.
- Air risk: no airspace collision-risk map is available and no Annex C (local air-density) claim is made. Assume the operator claims any reduction the SORA 2.5 Main Body itself grants for the way the operation is flown; do not assume any other strategic mitigation or agreement.
- Otherwise apply SORA 2.5 as written.
- If the operation is outside the scope of SORA (e.g. a "not part of SORA" cell, or a final GRC above 7), report "sail": "certified" (and "igrc": "out_of_scope" if the iGRC cell itself is not part of SORA).

Work through the steps, then end your answer with a JSON object in a ```json code block with exactly these keys:
{"igrc": <integer or "out_of_scope">, "final_grc": <integer or null>, "initial_arc": "a|b|c|d", "residual_arc": "a|b|c|d", "tmpr": "none|low|medium|high", "sail": "I|II|III|IV|V|VI|certified"}
("tmpr" is the TMPR the operation must meet; "none" if no TMPR is required.)"""

SYSTEM_TR = """Bir havacılık emniyeti uzmanısınız. Kullanıcının tanımladığı İHA operasyonu için JARUS SORA 2.5 değerlendirmesi yapın (Specific Operations Risk Assessment / Özel Operasyon Risk Değerlendirmesi, sürüm 2.5, JARUS 2024). Şunları belirleyin: içsel yer riski sınıfı (iGRC, Adım 2), nihai GRC (Adım 3), başlangıç hava riski sınıfı (ARC, Adım 4), artık ARC (Adım 5), taktik azaltım performans gereksinimi (TMPR, Adım 6) ve SAIL (Adım 7).

Bu görevin kuralları:
- Yalnızca işleticinin açıkça talep ettiği yer riski azaltımlarını dikkate alın. Talep edilen her azaltım için belirtilen sağlamlık seviyesini (veya bütünlük ve güvence seviyelerini) otoritece kanıtlanmış ve kabul edilmiş sayın; kanıtları yeniden değerlendirmeyin.
- Hava riski: hava sahası çarpışma riski haritası yoktur ve Ek C (yerel hava trafiği yoğunluğu) kapsamında bir talep yapılmamaktadır. SORA 2.5 Ana Metni'nin operasyonun uçuş şekli için doğrudan tanıdığı her türlü düşürmenin işletici tarafından talep edildiğini varsayın; bunun dışında hiçbir stratejik azaltım veya mutabakat varsaymayın.
- Bunun dışında SORA 2.5'i yazıldığı gibi uygulayın.
- Operasyon SORA kapsamı dışındaysa (ör. "SORA'nın parçası değil" hücresi veya 7'den büyük nihai GRC), "sail": "certified" yazın (iGRC hücresi SORA kapsamı dışındaysa ayrıca "igrc": "out_of_scope").

Adımları çözün ve cevabınızı ```json kod bloğu içinde, tam olarak şu anahtarlara sahip bir JSON nesnesiyle bitirin:
{"igrc": <tam sayı veya "out_of_scope">, "final_grc": <tam sayı veya null>, "initial_arc": "a|b|c|d", "residual_arc": "a|b|c|d", "tmpr": "none|low|medium|high", "sail": "I|II|III|IV|V|VI|certified"}
("tmpr", operasyonun karşılaması gereken TMPR'dir; TMPR gerekmiyorsa "none".)"""

REFERENCE_EN = """SORA 2.5 REFERENCE CARD (JARUS SORA Main Body 2.5, 2024; restated)

STEP 2 - iGRC
- Column = the left-most column whose maximum characteristic dimension AND maximum speed both cover the UA:
  C1 <=1 m and <=25 m/s | C2 <=3 m and <=35 m/s | C3 <=8 m and <=75 m/s | C4 <=20 m and <=120 m/s | C5 <=40 m and <=200 m/s.
  Maximum speed = the designer's maximum commanded airspeed, not the mission speed.
- Row = the HIGHEST population density anywhere in the iGRC footprint (operational volume + ground risk buffer).
- iGRC values (C1 C2 C3 C4 C5):
  Controlled ground area: 1 1 2 3 3
  < 5 people/km2:         2 3 4 5 6
  < 50:                   3 4 5 6 7
  < 500:                  4 5 6 7 8
  < 5,000:                5 6 7 8 9
  < 50,000:               6 7 8 9 10
  > 50,000:               7 8 x x x   (x = not part of SORA -> certified category)
- A UA of 250 g or less with a maximum speed of 25 m/s or less has iGRC 1 regardless of population density.
- Qualitative descriptors (when no population map is used):
  controlled ground area = only persons involved in the operation are present;
  < 5 = remote: forests, deserts, large farm parcels, about one small building per km2;
  < 50 = lightly populated: small farms, residences on very large lots (about 16,000 m2);
  < 500 = sparsely populated: homes and small businesses on large lots (about 4,000 m2);
  < 5,000 = suburban / low-density metropolitan: single-family homes on small lots, apartment complexes, commercial buildings, mostly below 3-4 storeys;
  < 50,000 = high-density metropolitan: mostly large multistorey buildings, city downtown;
  > 50,000 = assemblies of people: large sporting events, large concerts.

STEP 3 - final GRC = iGRC + credits, applied in the order M1(A), M1(B), M1(C), M2
  Mitigation                          Low   Medium  High
  M1(A) sheltering                    -1    -2      N/A
  M1(B) operational restrictions      N/A   -1      -2
  M1(C) ground observation            -1    N/A     N/A
  M2 effects of impact dynamics       N/A   -1      -2
- N/A = no credit at that robustness. Robustness = the LOWER of the integrity level and the assurance level.
- M1(A) at medium robustness cannot be combined with any M1(B) mitigation.
- After all M1 credits, the GRC cannot be lower than the column's controlled-ground-area value; M2 is applied afterwards.
- Final GRC above 7 -> outside SORA (certified category).

STEP 4 - initial ARC (use the vertical limit of the operational volume = top of flight geography + contingency volume)
  1. Atypical / segregated airspace -> ARC-a
  2. Above FL600 -> ARC-b
  3. Airport/heliport environment: class B, C or D airspace -> ARC-d; otherwise (E, F, G) -> ARC-c
  4. Above 500 ft AGL (below FL600): Mode-C veil or TMZ -> ARC-d; else controlled airspace -> ARC-d; else uncontrolled over urban -> ARC-c; else uncontrolled over rural -> ARC-c
  5. Below 500 ft AGL: Mode-C veil or TMZ -> ARC-c; else controlled airspace -> ARC-c; else uncontrolled over urban -> ARC-c; else uncontrolled over rural -> ARC-b

STEP 5 - residual ARC
- VLOS, or BVLOS with an airspace observer alongside the remote pilot, lowers the initial ARC by one class, but never to ARC-a.
- Otherwise residual ARC = initial ARC (no other strategic mitigation is claimed in this task).

STEP 6 - TMPR
- VLOS: no TMPR (VLOS is an acceptable tactical mitigation for all ARCs).
- BVLOS: ARC-a none, ARC-b low, ARC-c medium, ARC-d high.

STEP 7 - SAIL (final GRC vs residual ARC a / b / c / d)
  <=2: I   II  IV VI
  3:   II  II  IV VI
  4:   III III IV VI
  5:   IV  IV  IV VI
  6:   V   V   V  VI
  7:   VI  VI  VI VI
  >7:  certified category"""

REFERENCE_TR = """SORA 2.5 BAŞVURU KARTI (JARUS SORA Ana Metin 2.5, 2024; yeniden ifade edilmiştir)

ADIM 2 - iGRC
- Sütun = İHA'nın hem azami karakteristik boyutunu HEM DE azami hızını karşılayan en soldaki sütun:
  S1 <=1 m ve <=25 m/s | S2 <=3 m ve <=35 m/s | S3 <=8 m ve <=75 m/s | S4 <=20 m ve <=120 m/s | S5 <=40 m ve <=200 m/s.
  Azami hız = tasarımcının beyan ettiği azami komut edilebilir hava hızıdır, görev hızı değildir.
- Satır = iGRC izdüşümünün (operasyon hacmi + yer riski tampon bölgesi) herhangi bir yerindeki EN YÜKSEK nüfus yoğunluğu.
- iGRC değerleri (S1 S2 S3 S4 S5):
  Kontrollü yer alanı:  1 1 2 3 3
  < 5 kişi/km2:         2 3 4 5 6
  < 50:                 3 4 5 6 7
  < 500:                4 5 6 7 8
  < 5.000:              5 6 7 8 9
  < 50.000:             6 7 8 9 10
  > 50.000:             7 8 x x x   (x = SORA'nın parçası değil -> sertifikalı kategori)
- 250 g veya daha hafif ve azami hızı 25 m/s veya daha düşük bir İHA, nüfus yoğunluğundan bağımsız olarak iGRC 1 alır.
- Niteliksel tanımlayıcılar (nüfus haritası kullanılmadığında):
  kontrollü yer alanı = yalnızca operasyona dahil kişiler bulunur;
  < 5 = ücra: ormanlar, çöller, geniş tarım parselleri, km2 başına yaklaşık bir küçük yapı;
  < 50 = çok seyrek nüfuslu: küçük çiftlikler, çok büyük parsellerde konutlar (yaklaşık 16.000 m2);
  < 500 = seyrek nüfuslu: geniş parsellerde (yaklaşık 4.000 m2) evler ve küçük işletmeler;
  < 5.000 = banliyö / düşük yoğunluklu kentsel: küçük parsellerde müstakil evler, apartman siteleri, ticari binalar, çoğu 3-4 katın altında;
  < 50.000 = yüksek yoğunluklu kentsel: ağırlıklı olarak büyük çok katlı binalar, şehir merkezi;
  > 50.000 = insan toplulukları: büyük spor etkinlikleri, büyük konserler.

ADIM 3 - nihai GRC = iGRC + krediler; sırasıyla M1(A), M1(B), M1(C), M2 uygulanır
  Azaltım                                   Düşük  Orta  Yüksek
  M1(A) barınma                             -1     -2    N/A
  M1(B) operasyonel kısıtlamalar            N/A    -1    -2
  M1(C) yer gözlemi                         -1     N/A   N/A
  M2 çarpma dinamiklerinin etkileri         N/A    -1    -2
- N/A = o sağlamlık seviyesinde kredi yoktur. Sağlamlık = bütünlük ve güvence seviyelerinden DÜŞÜK olanı.
- Orta sağlamlıktaki M1(A), herhangi bir M1(B) azaltımıyla birleştirilemez.
- Tüm M1 kredilerinden sonra GRC, ilgili sütunun kontrollü yer alanı değerinin altına inemez; M2 bundan sonra uygulanır.
- 7'den büyük nihai GRC -> SORA kapsamı dışı (sertifikalı kategori).

ADIM 4 - başlangıç ARC (operasyon hacminin dikey sınırını kullanın = uçuş coğrafyasının üst sınırı + acil durum hacmi)
  1. Atipik / ayrılmış (segregated) hava sahası -> ARC-a
  2. FL600 üzeri -> ARC-b
  3. Havaalanı/heliport ortamı: B, C veya D sınıfı hava sahası -> ARC-d; aksi halde (E, F, G) -> ARC-c
  4. 500 ft AGL üzeri (FL600 altı): Mode-C veil veya TMZ -> ARC-d; değilse kontrollü hava sahası -> ARC-d; değilse kentsel alan üzerinde kontrolsüz -> ARC-c; değilse kırsal alan üzerinde kontrolsüz -> ARC-c
  5. 500 ft AGL altı: Mode-C veil veya TMZ -> ARC-c; değilse kontrollü hava sahası -> ARC-c; değilse kentsel üzerinde kontrolsüz -> ARC-c; değilse kırsal üzerinde kontrolsüz -> ARC-b

ADIM 5 - artık ARC
- VLOS veya uzaktan pilotun yanında bir hava sahası gözlemcisi bulunan BVLOS, başlangıç ARC'yi bir sınıf düşürür; ancak asla ARC-a'ya indirmez.
- Aksi halde artık ARC = başlangıç ARC (bu görevde başka stratejik azaltım talep edilmemektedir).

ADIM 6 - TMPR
- VLOS: TMPR yok (VLOS tüm ARC'ler için kabul edilebilir taktik azaltımdır).
- BVLOS: ARC-a yok, ARC-b düşük (low), ARC-c orta (medium), ARC-d yüksek (high).

ADIM 7 - SAIL (nihai GRC ve artık ARC a / b / c / d)
  <=2: I   II  IV VI
  3:   II  II  IV VI
  4:   III III IV VI
  5:   IV  IV  IV VI
  6:   V   V   V  VI
  7:   VI  VI  VI VI
  >7:  sertifikalı kategori"""

EXTRACT_EN = """You are assisting with a JARUS SORA 2.5 assessment. A deterministic, validated SORA calculator will compute the risk classes; your only job is to extract its input parameters from the operation description provided by the user. Do not compute any risk class.

Return only a JSON object in a ```json code block with exactly these keys:
{
 "mtom_kg": number,                       // maximum take-off mass in kg
 "max_dimension_m": number,               // maximum characteristic dimension (wingspan, rotor diameter, or max distance between blade tips)
 "max_speed_ms": number,                  // designer-declared maximum speed in m/s (not the mission or cruise speed)
 "controlled_ground_area": true|false,    // only persons involved in the operation are present on the ground
 "max_population_density": number|null,   // HIGHEST population density (people/km2) anywhere in the footprint, when a number is given; else null
 "population_band": "controlled"|"<5"|"<50"|"<500"|"<5000"|"<50000"|">50000",   // band of that highest density; for qualitative descriptions use the table below
 "mitigations": {                         // each: null if not claimed, else {"integrity": "low|medium|high", "assurance": "low|medium|high"}; if only one robustness level is stated, use it for both
   "m1a": ..., "m1b": ..., "m1c": ..., "m2": ...
 },
 "atypical_airspace": true|false,         // atypical / segregated airspace
 "above_fl600": true|false,
 "airport_environment": true|false,       // operation in an airport/heliport environment
 "airport_airspace_class": "A"|"B"|"C"|"D"|"E"|"F"|"G"|null,   // class of the airspace in the airport environment, else null
 "flight_geography_top_m_agl": number,    // top of the flight geography, metres above ground (use 0 if the operation is above FL600)
 "contingency_above_m": number,           // how far the contingency volume extends above the flight geography (0 if none)
 "tmz_or_mode_c_veil": true|false,
 "controlled_airspace": true|false,       // controlled airspace (class A-E) outside an airport environment
 "urban": true|false,                     // overflown area is urban
 "vlos": true|false,
 "airspace_observer_alongside_pilot": true|false
}

Qualitative population descriptors: controlled ground area = only involved persons present; <5 = forests, deserts, large farm parcels, about one small building per km2; <50 = small farms, residences on very large lots (about 16,000 m2); <500 = homes and small businesses on large lots (about 4,000 m2); <5000 = single-family homes on small lots, apartment complexes, commercial buildings, mostly below 3-4 storeys; <50000 = mostly large multistorey buildings, city downtown; >50000 = assemblies of people such as large sporting events or concerts."""

EXTRACT_TR = """Bir JARUS SORA 2.5 değerlendirmesine yardımcı oluyorsunuz. Risk sınıflarını deterministik, doğrulanmış bir SORA hesaplayıcısı hesaplayacaktır; sizin tek göreviniz kullanıcının verdiği operasyon tanımından hesaplayıcının girdi parametrelerini çıkarmaktır. Hiçbir risk sınıfını hesaplamayın.

Yalnızca ```json kod bloğu içinde, tam olarak şu anahtarlara sahip bir JSON nesnesi döndürün:
{
 "mtom_kg": sayı,                          // azami kalkış kütlesi, kg
 "max_dimension_m": sayı,                  // azami karakteristik boyut (kanat açıklığı, rotor çapı veya pervane uçları arasındaki en büyük mesafe)
 "max_speed_ms": sayı,                     // tasarımcının beyan ettiği azami hız, m/s (görev veya seyir hızı değil)
 "controlled_ground_area": true|false,     // yerde yalnızca operasyona dahil kişiler bulunur
 "max_population_density": sayı|null,      // izdüşümün herhangi bir yerindeki EN YÜKSEK nüfus yoğunluğu (kişi/km2), sayı verilmişse; yoksa null
 "population_band": "controlled"|"<5"|"<50"|"<500"|"<5000"|"<50000"|">50000",   // bu en yüksek yoğunluğun bandı; niteliksel tanımlarda aşağıdaki tabloyu kullanın
 "mitigations": {                          // her biri: talep edilmemişse null, aksi halde {"integrity": "low|medium|high", "assurance": "low|medium|high"}; yalnızca tek bir sağlamlık seviyesi verilmişse ikisi için de onu kullanın
   "m1a": ..., "m1b": ..., "m1c": ..., "m2": ...
 },
 "atypical_airspace": true|false,          // atipik / ayrılmış (segregated) hava sahası
 "above_fl600": true|false,
 "airport_environment": true|false,        // havaalanı/heliport ortamında operasyon
 "airport_airspace_class": "A"|"B"|"C"|"D"|"E"|"F"|"G"|null,   // havaalanı ortamındaki hava sahası sınıfı, yoksa null
 "flight_geography_top_m_agl": sayı,       // uçuş coğrafyasının üst sınırı, yerden metre (operasyon FL600 üzerindeyse 0)
 "contingency_above_m": sayı,              // acil durum hacminin uçuş coğrafyasının ne kadar üzerine çıktığı (yoksa 0)
 "tmz_or_mode_c_veil": true|false,
 "controlled_airspace": true|false,        // havaalanı ortamı dışında kontrollü hava sahası (A-E sınıfı)
 "urban": true|false,                      // üzerinden uçulan bölge kentsel
 "vlos": true|false,
 "airspace_observer_alongside_pilot": true|false
}
Değerler için İngilizce anahtar kelimeleri kullanın (low/medium/high, true/false).

Niteliksel nüfus tanımlayıcıları: kontrollü yer alanı = yalnızca operasyona dahil kişiler; <5 = ormanlar, çöller, geniş tarım parselleri, km2 başına yaklaşık bir küçük yapı; <50 = küçük çiftlikler, çok büyük parsellerde konutlar (yaklaşık 16.000 m2); <500 = geniş parsellerde (yaklaşık 4.000 m2) evler ve küçük işletmeler; <5000 = küçük parsellerde müstakil evler, apartman siteleri, ticari binalar, çoğu 3-4 katın altında; <50000 = ağırlıklı olarak büyük çok katlı binalar, şehir merkezi; >50000 = büyük spor etkinlikleri veya konserler gibi insan toplulukları."""

USER_EN = "Operation description:\n\n{text}"
USER_TR = "Operasyon tanımı:\n\n{text}"

CONDITIONS = ("closed_book", "reference", "extract")


def build(condition: str, lang: str, text: str):
    """Return (system_blocks, user_text). system_blocks is a list of strings; the last block is cacheable."""
    if condition == "closed_book":
        system = [SYSTEM_EN if lang == "en" else SYSTEM_TR]
    elif condition == "reference":
        system = [SYSTEM_EN if lang == "en" else SYSTEM_TR, REFERENCE_EN if lang == "en" else REFERENCE_TR]
    elif condition == "extract":
        system = [EXTRACT_EN if lang == "en" else EXTRACT_TR]
    else:
        raise ValueError(condition)
    user = (USER_EN if lang == "en" else USER_TR).format(text=text)
    return system, user
