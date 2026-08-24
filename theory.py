# -*- coding: utf-8 -*-
"""
理論知識庫 (Theory Knowledge Base)
──────────────────────────────────
為系統中每個工具／統計方法／機器學習演算法提供：
原理、前提假設、適用場景、不適用陷阱、實務建議、參考資料。

使用方式（在任何 tab 或聊天流程中一行接線）：

    from theory import render_theory
    render_theory("ttest")          # 統計檢定
    render_theory("random_forest")  # 機器學習模型
    render_theory("psi")            # 監控指標

渲染效果：一個「📖 理論與適用場景」小按鈕（st.popover），
點開後顯示結構化說明，不佔用版面空間。
"""

import streamlit as st

# ══════════════════════════════════════════════════════════════
# 知識本體
# 欄位：name / one_liner / theory / assumptions /
#       good_for(list) / not_good_for(list) / tips / reference
# ══════════════════════════════════════════════════════════════
THEORY_KB = {

    # ─────────────────────────────────────────────
    # 統計分析
    # ─────────────────────────────────────────────
    "descriptive_stats": {
        "name": "敘述統計 (Descriptive Statistics)",
        "one_liner": "EDA 的第一步：用少量指標快速掌握每個變數的「樣貌」。",
        "theory": (
            "以平均數、中位數、標準差、四分位距、最大最小值等指標摘要資料分布。"
            "**平均數**對極端值敏感，**中位數**穩健——兩者差距大即暗示偏態或離群值。"
            "類別欄位則看眾數、唯一值數與各類別占比。"
        ),
        "good_for": [
            "建模前的資料體檢（缺失、量級、偏態一次掌握）",
            "跨欄位比較量級 → 判斷是否需要標準化",
            "向非技術利害關係人報告資料現況",
        ],
        "not_good_for": [
            "推論因果關係（只是描述，不是檢定）",
            "高度偏態資料只看平均數會誤判中心位置",
        ],
        "tips": [
            "mean 與 median 差距 >30% 時，優先檢查離群值再決定補值策略",
            "std 接近 0 的欄位是常數欄，建模前應剔除",
        ],
        "reference": "[Tukey (1977) EDA](https://en.wikipedia.org/wiki/Exploratory_data_analysis)",
    },

    "ttest": {
        "name": "t 檢定 (t-Test)",
        "one_liner": "比較兩組數值的平均值是否有統計顯著差異。",
        "theory": (
            "以兩組均值差除以合併標準誤得到 t 統計量，在虛無假設「兩組均值相同」下"
            "服從 t 分布。P < α 時拒絕虛無假設。"
            "**本系統已內建自動前提檢查**：Shapiro-Wilk 檢常態性（不通過→自動改用 "
            "Mann-Whitney U）；Levene 檢變異數同質性（不通過→改用 Welch's t）。"
        ),
        "assumptions": [
            "觀察值獨立（無重複測量）",
            "各組資料近似常態（n>30 時由中央極限定理緩解）",
            "Student t 需變異數相等；Welch t 不需要",
        ],
        "good_for": [
            "A/B 測試成效比較",
            "比較二元目標下連續特徵的組間差異（如：違約 vs 未違約者的收入）",
            "實驗前後的指標變化檢定",
        ],
        "not_good_for": [
            "三組以上比較（應用 ANOVA，多次 t 檢定會膨脹第一型錯誤）",
            "嚴重偏態或有離群值的資料（自動改用 Mann-Whitney U）",
            "配對樣本（同一主體前後測，需成對 t 檢定）",
        ],
        "tips": [
            "報告結果時附上效果量 Cohen's d：p 顯著但 d<0.2 代表差異微小、缺乏實務意義",
            "大樣本下微不足道的差異也會顯著 → 一定要看效果量",
        ],
        "reference": "[Student's t-test](https://en.wikipedia.org/wiki/Student%27s_t-test)",
    },

    "mannwhitney": {
        "name": "Mann-Whitney U 檢定（無母數）",
        "one_liner": "不假設常態的兩組比較：比的是「分布位置（中位數）」而非均值。",
        "theory": (
            "把兩組觀察值混合排序，U 統計量衡量一組值普遍大於另一組的程度。"
            "對離群值與偏態穩健，是 t 檢定常態性不符時的標準替代方案。"
            "**本系統在 t 檢定的 Shapiro-Wilk 未通過時會自動切換至此方法。**"
        ),
        "good_for": [
            "所得、保額等右偏資料的組間比較",
            "序位尺度資料（李克特量表等）",
            "小樣本且無法驗證常態時",
        ],
        "not_good_for": [
            "需要直接比較「平均值」的場景（它檢定的是分布位置）",
            "配對樣本（應用 Wilcoxon 符號等級檢定）",
        ],
        "tips": ["搭配 rank-biserial 效果量一起報告"],
    },

    "anova": {
        "name": "單因子變異數分析 (One-Way ANOVA)",
        "one_liner": "一次比較三組以上的平均值差異。",
        "theory": (
            "將總變異拆成「組間變異」與「組內變異」，F = 組間均方 / 組內均方。"
            "F 顯著代表至少有一組不同，但不知道是哪幾組 → 需要 Tukey HSD 事後成對比較。"
            "效果量 η² 表示分組因素解釋了多少比例的變異。"
            "**本系統自動前置檢查**：多數組常態性未過 → 自動改用 Kruskal-Wallis。"
        ),
        "assumptions": [
            "各組獨立且近似常態",
            "變異數同質（Levene 檢定；本系統會自動檢查並提示）",
        ],
        "good_for": [
            "比較多個通路／區域／產品線的績效指標",
            "類別特徵與連續目標的關聯初探（f_classif 同源）",
        ],
        "not_good_for": [
            "僅兩組（用 t 檢定更直接）",
            "非常態多組比較（自動改用 Kruskal-Wallis）",
            "重複測量設計",
        ],
        "tips": [
            "ANOVA 顯著後務必看 Tukey HSD 找出差異來源",
            "η² < 0.01 微弱 / <0.06 小 / <0.14 中 / 其餘大",
        ],
    },

    "kruskal": {
        "name": "Kruskal-Wallis H 檢定（無母數 ANOVA）",
        "one_liner": "多組版本的 Mann-Whitney U：不要求常態，比的是分布位置。",
        "theory": (
            "將所有組混合排序後，以組間秩和差異構成 H 統計量，近似 χ² 分布。"
            "事後成對比較用 Mann-Whitney U 加 Bonferroni 校正"
            "（α 除以成對數，控制多重檢定的家族錯誤率）。"
        ),
        "good_for": [
            "非常態、有離群值的多組比較",
            "順序型或多峰分布的群組差異",
        ],
        "not_good_for": [
            "需要估計組均值差的場景",
            "組內樣本數極少（<3）時檢定力很低",
        ],
    },

    "chi_square": {
        "name": "卡方獨立性檢定 (Chi-Square Test)",
        "one_liner": "檢定兩個類別變數之間是否有關聯。",
        "theory": (
            "建立列聯表，比較觀察次數 O 與「若獨立時的期望次數」E 的差距，"
            "χ² = Σ(O-E)²/E。P < α 代表兩變數不獨立（有關聯）。"
            "效果量 Cramér's V：<0.1 微弱 / <0.3 弱 / <0.5 中等 / ≥0.5 強。"
        ),
        "assumptions": [
            "觀察值獨立（一格一筆）",
            "期望次數 ≥5（低於時本系統提示，可考慮 Fisher 精確檢定或合併類別）",
        ],
        "good_for": [
            "類別 × 類別 特徵與目標的關聯篩選（如：地區 × 是否理賠）",
            "問卷交叉分析",
            "對應人工篩選流程中剔除不顯著類別變數的第一道關卡",
        ],
        "not_good_for": [
            "連續變數（先分箱或改用相關分析）",
            "期望次數過低的稀疏表格（χ² 會失真）",
            "顯著 ≠ 關係強：大樣本下微弱關聯也會顯著，必須看 Cramér's V",
        ],
        "tips": [
            "高基數類別（如郵遞區號）先合併稀有類別再檢定",
            "列聯表列/欄過多時 χ² 自由度暴增、檢定力下降",
        ],
    },

    "correlation": {
        "name": "相關分析 (Correlation Analysis)",
        "one_liner": "量化兩個連續變數的線性／單調關聯強度與方向。",
        "theory": (
            "**Pearson r** 衡量線性關聯（假設近常態、對離群值敏感）；"
            "**Spearman ρ** 把資料轉成排名後算相關，只要求單調關係、較穩健；"
            "**Kendall τ** 基於協調對比例，小樣本下更穩定但計算慢。"
            "|r|≥0.7 非常強、0.5~0.7 強、0.3~0.5 中等。"
        ),
        "good_for": [
            "特徵共線性偵測（|r|>0.9 的配對建議擇一入模）",
            "特徵與目標的初步關聯掃描",
            "散點圖前的量化排序（決定優先畫哪些）",
        ],
        "not_good_for": [
            "非線性關聯（U 型關係 Pearson r 可能≈0）→ 先畫散點圖",
            "離群值敏感：一個極端值就能拉高 Pearson r",
            "相關 ≠ 因果",
        ],
        "tips": [
            "偏態資料優先用 Spearman",
            "高相關配對擇優請搭配樹模型重要性（見「特徵海選」）",
        ],
    },

    "linear_regression_ols": {
        "name": "線性迴歸 (OLS)",
        "one_liner": "用直線（超平面）描述目標與特徵的平均關係，係數可直接詮釋。",
        "theory": (
            "以最小平方法估計 y = β₀ + β₁x₁ + … + ε，使殘差平方和最小。"
            "statsmodels 摘要提供係數、標準誤、p 值、R²、F 檢定等完整診斷。"
        ),
        "assumptions": [
            "線性關係、殘差常態且同質、無嚴重多元共線性（VIF>10 警戒）",
        ],
        "good_for": [
            "需要向業務解釋「每增加一單位 x，y 平均變化多少」",
            "快速建立迴歸 baseline",
            "特徵顯著性的統計推論",
        ],
        "not_good_for": [
            "非線性關係（先做轉換或改用樹模型）",
            "共線性高的特徵群（係數會互相干擾、符號反轉）",
            "分類目標（用 Logistic Regression）",
        ],
        "tips": ["先跑 VIF 剔除共線變數，係數才可信"],
    },

    "woe_iv": {
        "name": "WOE / IV 分析 (Weight of Evidence & Information Value)",
        "one_liner": "信用評分實務的變數篩選黃金標準：量化每個變數對二元目標的預測力。",
        "theory": (
            "將連續變數分箱後，計算每箱的好壞客戶分布比：WOE = ln(壞件占比 / 好件占比)。"
            "IV = Σ(壞件占比−好件占比)×WOE，加總衡量整體預測力。\n\n"
            "**IV 判讀**：<0.02 無預測力（剔除）｜0.02~0.1 弱｜0.1~0.3 中｜"
            "0.3~0.5 強｜**≥0.5 極強（警惕目標洩漏！）**"
        ),
        "good_for": [
            "信用風險、保險理賠等二元目標的特徵篩選（對應人工流程第一道海選）",
            "為評分卡模型準備 WOE 轉換後的變數",
            "向風管主管解釋每個變數為何入模",
        ],
        "not_good_for": [
            "多元分類或連續目標",
            "IV≥0.5 的變數直接入模——極可能含目標資訊（洩漏），務必人工確認業務邏輯",
            "稀有類別過多的箱（WOE 波動劇烈）",
        ],
        "tips": [
            "分箱方法：等頻穩健、決策樹切點最有判別力、等寬易受極端值扭曲",
            "WOE 曲線應大致單調；鋸齒狀代表分箱不穩，需合併相鄰箱",
        ],
        "reference": "[Siddiqi (2006) Credit Risk Scorecards](https://en.wikipedia.org/wiki/Information_value)",
    },

    "vif": {
        "name": "VIF 變異數膨脹因子",
        "one_liner": "診斷多元共線性：某變數被其他特徵線性合成的程度。",
        "theory": (
            "VIF_j = 1/(1−R²j)，R²j 是「其他特徵回歸第 j 個特徵」的判定係數。"
            "VIF>10 嚴重共線、5~10 需留意、<5 可接受。"
        ),
        "good_for": ["OLS 係數解讀前的必要健檢", "找出冗餘特徵對"],
        "tips": [
            "逐次剔除 VIF 最高者並重算，直到全部 <10",
            "One-Hot 後的全類別組會天然共線（虛擬變數陷阱），留 k−1 類",
        ],
    },

    # ─────────────────────────────────────────────
    # 特徵篩選
    # ─────────────────────────────────────────────
    "univariate_screen": {
        "name": "單變量統計篩選 (Univariate Screening)",
        "one_liner": "逐一檢定每個特徵與目標的關聯，剔除 p > 0.05 的無訊息變數。",
        "theory": (
            "數值特徵＋分類目標 → f_classif（ANOVA F 檢定）；"
            "數值特徵＋連續目標 → f_regression；類別特徵 → 卡方檢定。"
            "對應人工流程「全變數卡方迴圈、p>0.05 剔除」的自動化版本。"
        ),
        "good_for": [
            "高維資料的第一道快速過濾（計算成本極低）",
            "剔除明顯與目標無關的噪聲特徵，減輕後續步驟負擔",
        ],
        "not_good_for": [
            "只看單變量會漏掉交互作用項（弱特徵組合後可能變強）",
            "p 值在大樣本下過度寬鬆——保留清單仍需後續步驟收斂",
        ],
        "tips": ["建議流程：單變量 → 高相關擇優 → RFE/AIC，逐層收斂"],
    },

    "corr_prune": {
        "name": "高相關配對擇優 (Correlation Pruning)",
        "one_liner": "|r|>0.9 的變數對只留一個——依樹模型重要性保住較有用的那個。",
        "theory": (
            "計算特徵相關矩陣取上三角，找 |r| ≥ 門檻的配對；"
            "再用隨機森林重要性為每個配對打分，保留重要性高者。"
            "完全複刻人工筆記 Feature_select_model 的核心邏輯。"
        ),
        "good_for": [
            "消除共線性帶來的係數不穩與重要度稀釋",
            "降低維度而不損失資訊（冗餘特徵本質上是複本）",
        ],
        "not_good_for": [
            "閾值太低（<0.8）會誤刪互補特徵",
            "非線性冗餘偵測不到（Pearson 只看線性）",
        ],
        "tips": ["剔除清單務必人工過目：某些變數雖然冗餘但有法遵或業務意義"],
    },

    "rfe": {
        "name": "RFE 遞歸特徵消除 (Recursive Feature Elimination)",
        "one_liner": "反覆訓練模型、淘汰最不重要特徵，直到剩下指定數量。",
        "theory": (
            "以基學習器（本系統用 balanced Logistic Regression）在全特徵上訓練，"
            "依係數/重要性移除最弱者，重複至 n_features_to_select。"
            "屬於包裹式（wrapper）方法，考量了特徵之間的交互作用。"
        ),
        "good_for": [
            "需要「剛好 N 個特徵」的場景（部署成本受限時）",
            "單變量與相關篩選後的精緻收斂",
        ],
        "not_good_for": [
            "特徵數很大時速度慢（O(特徵數) 次訓練）",
            "基學習器偏誤會傳導到選取結果（LR 偏愛線性訊號）",
        ],
        "tips": ["搭配 cross-validated RFE（RFECV）可自動找最佳特徵數"],
    },

    "aic_stepwise": {
        "name": "AIC 逐步選擇 (Stepwise Selection)",
        "one_liner": "貪婪加入／回剔讓 AIC 最小的變數組合，兼顧擬合與簡潔。",
        "theory": (
            "AIC = −2ln(L) + 2k，懲罰參數數量 k。逐步法每一步挑出使 AIC 下降最多的"
            "變數加入（forward），both 模式允許回剔已入選變數。"
            "**本系統用貪婪法取代暴力窮舉**：人工筆記 15 變數窮舉約需 40 分鐘，"
            "貪婪法 < 5 秒，結果通常相近。"
        ),
        "good_for": [
            "統計建模情境需要精簡且可解釋的變數集",
            "與 RFE 結果交叉驗證（兩法都留下的變數最穩）",
        ],
        "not_good_for": [
            "純預測任務（樹模型+重要性通常更好）",
            "貪婪法不保證全域最佳；變數高度相關時路徑依賴初始順序",
        ],
        "tips": ["AIC 下降 <2 視為無實質改善，及時停止"],
    },

    # ─────────────────────────────────────────────
    # 監控
    # ─────────────────────────────────────────────
    "psi": {
        "name": "PSI 族群穩定性指標 (Population Stability Index)",
        "one_liner": "量化「上線後的資料」相對「建模時資料」的偏移程度。",
        "theory": (
            "將兩期資料切成相同的箱，PSI = Σ(A占比−B占比)×ln(A占比/B占比)。"
            "**判讀**：<0.1 穩定｜0.1~0.25 中度偏移（密切觀察）｜"
            ">0.25 劇烈偏移（模型需重訓或重建）"
        ),
        "good_for": [
            "模型上線後的定期健康檢查（月/季）",
            "定位是哪個特徵漂移（per-feature PSI）",
            "疫情、政策變動後評估模型是否失效",
        ],
        "not_good_for": [
            "樣本數太少時箱占比波動會造成 PSI 虛高",
            "只能偵測「分布漂移」，不能直接反映預測準確度衰退",
        ],
        "tips": [
            "分箱方式要與建模時一致，否則 PSI 不可比",
            "PSI 高但業務無變化時，先確認資料管線沒壞",
        ],
        "reference": "[PSI](https://en.wikipedia.org/wiki/Precision_and_recall) · [Siddiqi (2006)](https://archive.org/details/creditriskscorin0000sidd)",
    },

    "ks_statistic": {
        "name": "KS 統計量 (Kolmogorov-Smirnov)",
        "one_liner": "好壞客戶累積分布的最大間距——風控模型的經典判別力指標。",
        "theory": (
            "對分數排序後分箱，KS = max|好件累積占比 − 壞件累積占比|。"
            "**判讀**：>0.4 良好、>0.3 可用、<0.2 判別力弱。"
            "KS 出現的位置也是潛在切分點參考。"
        ),
        "good_for": ["評分卡/風控模型判別力評估", "與 AUC 互補（KS 看最大分離點）"],
        "not_good_for": ["多分類", "KS 高但尾部校準差的模型仍可能有業務問題"],
    },

    "lift_gain": {
        "name": "Lift / Gain 圖",
        "one_liner": "回答業務問題：「模型挑出的前 10% 名單，捕捉了多少目標事件？」",
        "theory": (
            "按預測機率降序分箱（decile）。Gain=前 k% 內捕捉的事件占總事件比；"
            "Lift=Gain ÷ 隨機基準(k%)。Lift=3 代表前 10% 名單的事件密度是隨機的 3 倍。"
        ),
        "good_for": [
            "催收/行銷名單優先級排序的效果溝通",
            "向業務展示「不用模型 vs 用模型」的成本效益",
        ],
        "not_good_for": ["單獨使用（需搭配 AUC/KS 看整體判別力）"],
    },

    # ─────────────────────────────────────────────
    # 前處理策略
    # ─────────────────────────────────────────────
    "impute_strategy": {
        "name": "遺漏值處理策略",
        "one_liner": "補值方法的選擇取決於缺失機制與變數分布，沒有萬用答案。",
        "theory": (
            "**平均數**：分布近常態時無害；**中位數**：偏態/離群下的穩健選擇；"
            "**眾數**：類別欄首選；**獨立類別 Missing**：當「缺失本身有資訊」"
            "（如客戶拒答收入往往與風險相關）時最佳；**捨棄欄位**：缺失率 >50%。"
        ),
        "good_for": [
            "MCAR/MAR 情境下的快速修復",
            "本系統會依「偏態/離群/類別占比」逐欄自動推薦策略",
        ],
        "not_good_for": [
            "MNAR（非隨機缺失）：單純補值會引入偏誤，建議加缺失旗標欄",
            "高缺失又高重要性的欄位直接刪列會損失大量樣本",
        ],
        "tips": [
            "一律在 train/test 切割後、只用訓練集統計量補值（避免洩漏）",
            "補值前後跑一次敘述統計比對，確認分布沒被大幅扭曲",
        ],
    },

    "imbalance_handling": {
        "name": "類別不平衡處理",
        "one_liner": "少數類 <20% 時，Accuracy 會騙人——先平衡、再看對的指標。",
        "theory": (
            "**class_weight='balanced'**：損失函數中給少數類更高權重（無資訊損失，首選）；"
            "**SMOTE/ADASYN**：合成少數類新樣本（過採樣，注意只在訓練折做）；"
            "**欠採樣**：犧牲多數類樣本換平衡（實戰筆記的 1:10 做法，適合超大樣本）；"
            "**SMOTETomek**：合成後清理交界雜訊。"
        ),
        "good_for": [
            "詐欺、理賠、違約等 1%~20% 事件的典型風控場景",
            "配合 AUC/F1/Recall/PR-AUC 評估（不要看 Accuracy）",
        ],
        "not_good_for": [
            "在測試集上做 SMOTE（會洩漏，務必只作用於訓練資料）",
            "不平衡但業務只在乎 top-K 精準度時，門檻調整可能比抽樣更有效",
        ],
        "tips": [
            "首選 class_weight；樣本量百萬級以上再考慮欠採樣",
            "閾值預設 0.5 對不平衡資料幾乎必錯，務必做閾值優化",
        ],
    },
}


# ══════════════════════════════════════════════════════════════
# 機器學習演算法條目
# ══════════════════════════════════════════════════════════════
def _ml_entry(name, one_liner, theory, good, bad, tips=None, ref=None):
    return {"name": name, "one_liner": one_liner, "theory": theory,
            "good_for": good, "not_good_for": bad, "tips": tips or [],
            "reference": ref}


_ML_ENTRIES = {
    "logistic_regression": _ml_entry(
        "Logistic Regression 邏輯迴歸",
        "分類問題的第一把尺：係數即勝算比的對數，完全可解釋。",
        "以 Sigmoid 函數將線性組合壓縮到 (0,1) 作為事件機率，用最大概似估計係數。"
        "係數 e^β 即該特徵每增一單位的勝算比 (Odds Ratio)。",
        ["風控評分卡的產業標準（搭配 WOE）", "需要向法遵/業務解釋每一分數來源",
         "快速 baseline，訓練成本近乎為零"],
        ["只能捕捉線性決策邊界", "共線性會讓係數失真（先跑 VIF/WOE）",
         "特徵量大且關係複雜時精度不及 boosting"],
        ["務必搭配 class_weight 處理不平衡", "C 值越小正則化越強"]),
    "decision_tree": _ml_entry(
        "Decision Tree 決策樹",
        "一連串 if-else 規則構成的樹，人類可直接閱讀。",
        "每次分裂選擇「最能降低不純度（Gini/Entropy）」的特徵與切點，遞迴生長。"
        "max_depth/min_samples_leaf 控制樹的大小以防過擬合。",
        ["需要白箱規則的場景（核貸否決理由）", "特徵交互作用的可視化探索"],
        ["單棵樹方差極大、容易過擬合（資料微調結構劇變）",
         "預測精度通常不及 ensemble 方法"],
        ["深度限 5~7 層先看主規則", "永遠優先考慮升級成 Random Forest"]),
    "random_forest": _ml_entry(
        "Random Forest 隨機森林",
        "數百棵去相關決策樹投票——穩定、免調參也能打的萬用解。",
        "Bagging：每棵樹用 bootstrap 抽樣的資料 + 每次分裂只看隨機子集特徵。"
        "平均掉單棵樹的高方差，附帶 OOB 與特徵重要性。",
        ["表格資料的穩健預設模型", "特徵重要性排序（供特徵篩選擇優）",
         "對離群值與量綱不敏感（不需標準化）"],
        ["大量淺層規則時不如 boosting 精準", "千棵以上深樹的推論延遲與記憶體成本"],
        ["n_estimators 越多越穩（100~300 通常足夠）",
         "重要性在高相關特徵間會互相稀釋——先做相關擇優"]),
    "svm": _ml_entry(
        "SVM 支援向量機",
        "找最大化類別間隔的超平面；kernel trick 處理非線性。",
        "只依賴少數支持向量，對高維小樣本有效。probability=True 時用 Platt scaling "
        "額外校準機率（較慢）。",
        ["中小樣本 + 高維特徵（文本向量）", "決策邊界清晰的問題"],
        ["大樣本（>5萬）訓練極慢", "對特徵尺度敏感（必須標準化）",
         "核函數與 C/gamma 調參成本高"],
        ["先試 LinearSVC 再考慮 RBF kernel"]),
    "linearsvc": _ml_entry(
        "LinearSVC 線性 SVM（校準版）",
        "線性核 SVM + 機率校準，大樣本的快速替代方案。",
        "liblinear/sag 最佳化的線性 SVM，外包 CalibratedClassifierCV(cv=3) "
        "輸出可用機率。dual='auto' 自動選求解模式。",
        ["大樣本文字分類", "想要 SVM 邊界但需要 predict_proba"],
        ["仍是線性邊界", "校準增加 3 倍訓練時間"]),
    "knn": _ml_entry(
        "KNN K-近鄰",
        "物以類聚：用最近 k 個鄰居的多數票分類。",
        "懶學習：不做訓練，預測時計算與所有樣本的距離取 k 近鄰投票。"
        "距離度量受特徵尺度影響極大。",
        ["小型資料集的快速原型", "決策邊界高度局部非線性"],
        ["大樣本預測慢（每次都要掃全表）", "維度詛咒：高維時距離失去意義",
         "對噪聲與不平衡敏感"],
        ["k 用奇數避免平手", "務必先標準化特徵"]),
    "naive_bayes": _ml_entry(
        "Naive Bayes 樸素貝葉斯",
        "假設特徵條件獨立的機率分類器，快而糙但常常意外好用。",
        "Bayes 定理 + 特徵條件獨立假設，逐特徵估計似然再相乘。GaussianNB 假設常態。",
        ["文字分類 baseline", "訓練速度要求極快的串流場景"],
        ["特徵相關性強時機率估計失真", "預測機率本身不可靠（排序尚可）"],
        []),
    "lda": _ml_entry(
        "LDA 線性判別分析",
        "同時做降維與分類：投影到「類間散布最大」的方向。",
        "假設各類共變異矩陣相同、服從常態，估計 Bayes 最優線性判別面。"
        "附帶將資料投影到 k−1 維的降維能力。",
        ["二分類快速 baseline（速度快於 LR）", "多分類的可解釋降維視覺化"],
        ["共變異矩陣不等時失效（改 QDA）", "同樣怕共線性與非常態"],
        []),
    "adaboost": _ml_entry(
        "AdaBoost 適應性提升",
        "老牌 boosting：後一棵樹專注修正前面分錯的樣本。",
        "迭代訓練淺樹（樹樁），提高被分錯樣本的權重，最後加權投票。",
        ["乾淨表格資料的快速 boosting baseline", "作為理解 GBM 家族的教學起點"],
        ["對離群值與噪聲標籤敏感（錯分樣本權重爆炸）",
         "精度通常不及 XGBoost/LightGBM"],
        []),
    "extra_trees": _ml_entry(
        "Extra Trees 極度隨機樹",
        "比 RF 更激進的隨機化：切點也隨機選，方差更低、速度更快。",
        "與 RF 差異：(1) 不 bootstrap，用全部樣本；(2) 分裂切點均勻隨機。"
        "以略增偏差換取更低方差與訓練速度。",
        ["需要比 RF 更快的 ensemble", "噪聲較大的資料（隨機化抑制過擬合）"],
        ["特徵重要性解讀性略遜 RF"],
        []),
    "mlp": _ml_entry(
        "MLP 類神經網路 (多層感知器)",
        "全連接神經網路的 sklearn 實作——表格資料上的深度學習入門。",
        "多層神經元 + 非線性激活反向傳播訓練。hidden_layer_sizes=(100,) 為單隱層。",
        ["特徵間複雜非線性交互", "資料量大、且 boosting 已達瓶頸時嘗試"],
        ["表格資料上通常打不過 XGBoost/LightGBM",
         "對特徵標準化與超參數（alpha、架構）敏感", "訓練時間不可控（lbfgs 除外）"],
        ["務必先 StandardScaler", "小資料先用 boosting，別急著上 NN"]),
    "xgboost": _ml_entry(
        "XGBoost",
        "梯度提升樹王者之一：正則化 + 二階優化 + 缺失值原生處理。",
        "循序添加決策樹，每棵擬合前面模型的梯度（殘差方向），"
        "目標函數含 L1/L2 正則項抑制過擬合；split finding 支援稀疏缺失方向學習。",
        ["表格資料競賽與工業界主力", "含缺失值的原始資料（不需預先補值）",
         "特徵重要性與 SHAP 解釋生態完整"],
        ["超參數空間大（learning_rate/n_estimators/max_depth/subsample）",
         "小資料(<1千)容易過擬合，需早停與 CV"],
        ["先固定 learning_rate=0.1 調樹參數，再降 lr 加樹數",
         "scale_pos_weight ≈ 負/正樣本比 可處理不平衡"]),
    "lightgbm": _ml_entry(
        "LightGBM",
        "微軟系 boosting：直方圖加速 + 葉子優先生長，大資料的首選速度王。",
        "以 leaf-wise（最大增益葉分裂）取代 level-wise 生長，配合直方圖分桶與 "
        "GOSS/EFB 特征捆綁，訓練速度常為 XGBoost 的數倍。",
        ["十萬級以上樣本的快速訓練", "類別特徵原生支援（categorical_feature）"],
        ["leaf-wise 在小資料上容易過擬合（限制 num_leaves）",
         "深樹對噪聲敏感"],
        ["num_leaves 是首要調參點（31→63→127）", "min_data_in_leaf 防止過擬合"]),
    "catboost": _ml_entry(
        "CatBoost",
        "Yandex 系 boosting：類別特徵 target encoding 零設定、對稱樹抗噪。",
        "ordered target statistics 以時間序方式編碼類別特徵，避免目標洩漏；"
        "對稱 oblivious tree 結構增強泛化。",
        ["高基數類別特徵多的表格資料（免手工編碼）",
         "想要最少調參就有不錯效果的場景"],
        ["推理速度較慢（對稱樹深）", "小資料上未必贏過調參後的 LightGBM"],
        ["cat_features 參數直接傳欄位名即可", "預設 iterations=1000 配 early stopping"]),
    "gradient_boosting": _ml_entry(
        "Gradient Boosting (sklearn)",
        "boosting 概念的教科書實作——XGBoost 的前身。",
        "循序擬合殘差梯度，sklearn 版本功能較基礎但介面一致。",
        ["教學與概念驗證", "不想引入外部套件時的中型資料方案"],
        ["速度與精度均落後 XGBoost/LightGBM", "無原生缺失值處理"],
        []),
    "ridge": _ml_entry(
        "Ridge 嶺迴歸 (L2)",
        "線性迴歸 + L2 懲罰：係數縮小但不歸零，共線性救星。",
        "在最小平方目標加上 λΣβ²，迫使係數均勻縮小。alpha 越大懲罰越強。",
        ["特徵共線但全都想保留", "p 與 n 相近的高維線性問題"],
        ["不做特徵選擇（係數不會變 0）"],
        ["alpha 用 log scale 網格搜（0.001~100）"]),
    "lasso": _ml_entry(
        "Lasso 套索回歸 (L1)",
        "線性迴歸 + L1 懲罰：自動把沒用的係數壓到零＝內建特徵選擇。",
        "λΣ|β| 的絕對值懲罰使部分係數恰好為 0，輸出稀疏模型。",
        ["高維資料同時要建模+選特徵", "想獲得精簡變數清單的統計場景"],
        ["高度相關特徵中只任意保留一個", "n << p 時最多選 n 個特徵"],
        ["與 Ridge 疊加即 ElasticNet"]),
    "elasticnet": _ml_entry(
        "ElasticNet 彈性網路",
        "L1+L2 混合懲罰：Lasso 的選擇力 + Ridge 的穩定性。",
        "l1_ratio 控制 L1/L2 比例（1=純 Lasso，0=純 Ridge），其餘同前。",
        ["相關特徵群的稀疏建模（Lasso 只留一個的缺陷被修復）"],
        ["多一個 l1_ratio 要調"],
        []),
    "kmeans": _ml_entry(
        "K-Means K-平均分群",
        "把樣本分成 k 個球形簇——最快最常用的無監督分群。",
        "交替執行「分配到最近重心」與「重心更新」直至收斂；結果受初始質心影響，"
        "k-means++ 初始化與多次重跑 (n_init) 緩解。",
        ["客戶分層的快速原型", "特徵工程：距離最近重心作為新特徵"],
        ["必須事先指定 k（用 Elbow/Silhouette 選）",
         "只適合凸球狀簇；對環形/月牙形失效", "對離群值與尺度敏感（先標準化）"],
        ["k 的候選值跑 silhouette score 比較", "DBSCAN 適合不規則形狀"]),
    "dbscan": _ml_entry(
        "DBSCAN 密度分群",
        "不需要指定簇數：密度相連的點自成一群，離群點自動標 -1。",
        "以 eps（鄰域半徑）與 min_samples 定義核心點，密度可達區域擴張成簇。",
        ["形狀不規則的分群", "離群值偵測（label=-1 即異常點）"],
        ["eps/min_samples 難調且全局共用一套半徑",
         "各簇密度差異大時表現差", "高維空間距離失效"],
        ["用 k-距離圖找 eps 轉折點", "高維先 PCA 降維再分群"]),
}

# 合併進主知識庫
THEORY_KB.update(_ML_ENTRIES)

# ml_models.py 的顯示名稱 → 知識庫 key 對照
_MODEL_KEY_MAP = {
    "logistic regression": "logistic_regression",
    "decision tree": "decision_tree",
    "random forest": "random_forest",
    "svm": "svm", "linearsvc": "linearsvc",
    "knn": "knn",
    "naive bayes": "naive_bayes",
    "lda": "lda",
    "adaboost": "adaboost",
    "extra trees": "extra_trees",
    "mlp": "mlp",
    "xgboost": "xgboost",
    "lightgbm": "lightgbm",
    "catboost": "catboost",
    "gradient boosting": "gradient_boosting",
    "ridge": "ridge", "lasso": "lasso", "elasticnet": "elasticnet",
    "linear regression": "linear_regression_ols",
    "k-means": "kmeans", "dbscan": "dbscan",
}


def resolve_model_key(model_name: str):
    """將 ml_models.py 的顯示名稱解析為知識庫 key，找不到回 None。"""
    name_lower = str(model_name).lower()
    for frag, key in _MODEL_KEY_MAP.items():
        if name_lower.startswith(frag) or frag in name_lower:
            return key
    return None


# ══════════════════════════════════════════════════════════════
# 渲染元件
# ══════════════════════════════════════════════════════════════
_SECTIONS = [
    ("🔬 **原理**", "theory"),
    ("📏 **前提假設**", "assumptions"),
    ("✅ **適用場景**", "good_for"),
    ("⚠️ **不適用 / 陷阱**", "not_good_for"),
    ("💡 **實務建議**", "tips"),
]


def render_theory(key: str, label: str = "📖 理論與適用場景",
                  popover_label: str = None):
    """
    渲染理論說明小按鈕（st.popover，點開才展開內容）。
    key 不存在時靜默不渲染，方便放心呼叫。

    Args:
        key: 知識庫條目 key（如 "ttest"、"random_forest"、"psi"）
        label: 按鈕文字（預設 📖 理論與適用場景）
    """
    info = THEORY_KB.get(key)
    if not info:
        return
    try:
        pop = st.popover(label, help=None)
    except Exception:
        # 舊版 Streamlit fallback：expander
        pop = st.expander(label)

    with pop:
        st.markdown(f"#### {info.get('name', key)}")
        if info.get("one_liner"):
            st.caption(info["one_liner"])
        for title, field in _SECTIONS:
            val = info.get(field)
            if not val:
                continue
            st.markdown(title)
            if isinstance(val, (list, tuple)):
                for item in val:
                    st.markdown(f"- {item}")
            else:
                st.markdown(val)
        if info.get("reference"):
            st.markdown(f"📚 參考：{info['reference']}")


def render_theory_for_model(model_name: str):
    """便利包：直接吃 ml_models 的模型顯示名稱。"""
    key = resolve_model_key(model_name)
    if key:
        render_theory(key)


# 統計分析類型 → 知識庫 key（tab_statistics 用）
ANALYSIS_TYPE_KEYS = {
    "敘述統計": "descriptive_stats",
    "t 檢定": "ttest",
    "線性迴歸": "linear_regression_ols",
    "卡方檢定": "chi_square",
    "ANOVA 變異數分析": "anova",
    "相關分析": "correlation",
    "WOE/IV 分析": "woe_iv",
}

# 特徵海選步驟 → 知識庫 key（feature_selection 報告用）
FEATURE_SELECTION_KEYS = {
    "Step 1": "univariate_screen",
    "Step 2": "corr_prune",
    "Step 3": "rfe",
    "Step 4": "aic_stepwise",
}
