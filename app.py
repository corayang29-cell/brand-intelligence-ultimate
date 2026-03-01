import streamlit as st
import os
import re
import json
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Any, Set
from io import BytesIO
from datetime import date, datetime, timedelta
from groq import Groq

# ============================================================
# 🎯 Ultimate Brand Intelligence Platform with KOL Monitoring
# Campaign tracking, risk assessment, and automated reporting
# ============================================================

# Optional dependencies
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

try:
    import jieba
    import jieba.analyse
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False

# ============================================================
# 🔐 SECURE API KEY MANAGEMENT
# ============================================================

def get_groq_api_key() -> str:
    """Secure API key retrieval"""
    try:
        api_key = st.secrets.get("GROQ_API_KEY", "").strip()
        if api_key:
            return api_key
    except Exception:
        pass
    
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if api_key:
        return api_key
    
    return ""

def initialize_groq_client():
    """Initialize Groq client"""
    api_key = get_groq_api_key()
    
    if not api_key:
        st.sidebar.error("⚠️ Missing GROQ API KEY")
        st.sidebar.markdown("""
        **设置 API Key:**
        
        **方法1: Streamlit Secrets (推荐)**
        创建 `~/.streamlit/secrets.toml`:
        ```toml
        GROQ_API_KEY = "your-key"
        ```
        
        **方法2: 环境变量**
        ```bash
        export GROQ_API_KEY="your-key"
        ```
        """)
        
        temp_key = st.sidebar.text_input("临时输入 API Key", type="password")
        if temp_key:
            api_key = temp_key
        else:
            st.stop()
    
    try:
        client = Groq(api_key=api_key, timeout=30.0, max_retries=3)
        return client
    except Exception as e:
        st.error(f"❌ API初始化失败: {str(e)}")
        st.stop()

client = initialize_groq_client()

# ============================================================
# 📊 ENHANCED CONFIGURATION
# ============================================================

# Comprehensive stopwords
CHINESE_STOPWORDS = {
    # 通用停用词
    "的", "了", "是", "在", "我", "有", "和", "就", "不", "人", "都", "一", "个",
    "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好",
    "自己", "这", "那", "里", "来", "为", "但", "而", "与", "或", "啊", "呀", "吗",
    "呢", "吧", "哦", "哈", "嗯", "唉", "哎", "啦", "么", "嘛", "呗", "得",
    
    # 无意义高频词
    "真的", "感觉", "觉得", "还是", "然后", "所以", "因为", "如果", "这个", "那个",
    "什么", "怎么", "非常", "比较", "可能", "应该", "肯定", "一定", "绝对", "真是",
    "确实", "简直", "完全", "十分", "特别", "尤其", "总之", "反正", "其实", "本来",
    "已经", "还有", "而且", "不过", "只是", "可是", "但是", "虽然", "尽管",
    
    # 品牌相关（动态添加）
    "品牌", "牌子", "产品", "东西", "这款", "那款", "这个牌子", "那个牌子",
    "这家", "那家", "商家", "店家",
    
    # 电商平台
    "旗舰店", "官方", "店铺", "卖家", "买家", "客服", "购买", "下单", "收货",
    "包邮", "快递", "物流", "发货", "到货", "签收",
    
    # 小红书特色词
    "姐妹", "宝宝", "集美", "小红书", "笔记", "种草", "拔草", "入坑", "剁手",
    "安利", "分享", "推荐", "测评", "开箱", "晒单",
}

ENGLISH_STOPWORDS = {
    "the", "is", "are", "was", "were", "be", "been", "being", "a", "an", "and",
    "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "from", "as",
    "it", "this", "that", "these", "those", "i", "you", "he", "she", "we", "they",
    "my", "your", "his", "her", "our", "their", "what", "which", "who", "when",
    "where", "why", "how", "very", "really", "just", "so", "too", "more", "most"
}

# Enhanced sentiment terms
POSITIVE_TERMS = [
    # 产品体验
    "好", "棒", "喜欢", "爱", "推荐", "值得", "满意", "完美", "赞", "优秀", "不错",
    "好用", "实用", "舒服", "温和", "柔滑", "显白", "持久", "滋润", "保湿", "清爽",
    "轻薄", "服帖", "自然", "显色", "正", "顺滑", "细腻", "丝滑", "水润", "嫩",
    "舒适", "合适", "适合", "匹配", "贴合", "上妆", "妆感", "质感", "高级感",
    
    # 价值感知
    "划算", "超值", "惊喜", "物超所值", "性价比高", "实惠", "便宜", "优惠", "折扣",
    "白菜价", "良心", "亲民", "合理",
    
    # 外观包装
    "高级", "精致", "漂亮", "美", "好看", "颜值高", "大气", "时尚", "奢华", "档次",
    "质感", "有质感", "上档次", "讲究", "精美", "雅致",
    
    # 服务物流
    "快", "及时", "迅速", "热情", "耐心", "专业", "贴心", "周到", "满分",
    
    # 英文
    "good", "great", "love", "excellent", "perfect", "amazing", "best", "fantastic",
    "recommend", "worth", "satisfied", "quality", "premium", "smooth", "nice"
]

NEGATIVE_TERMS = [
    # 产品问题
    "差", "不好", "失望", "后悔", "避雷", "翻车", "坑", "垃圾", "难用", "鸡肋",
    "拔干", "卡纹", "掉色", "氧化", "暗沉", "显黑", "厚重", "油腻", "粘", "假白",
    "不持久", "易掉", "易花", "斑驳", "不均匀", "结块", "起皮", "过敏", "刺激",
    "不显色", "难推", "难卸", "浮粉", "卡粉", "脱妆", "晕染", "闷痘", "致痘",
    
    # 价格相关
    "不值", "贵", "贵了", "太贵", "溢价", "不划算", "性价比低", "坑钱", "智商税",
    "宰客", "黑心",
    
    # 真伪问题
    "假", "假货", "骗", "欺诈", "仿冒", "不是正品", "有问题", "怀疑", "盗版",
    "山寨", "仿品", "水货", "非正品",
    
    # 服务物流
    "退", "退货", "退款", "投诉", "维权", "拒退", "态度差", "不理", "慢", "延迟",
    "破损", "漏", "丢", "缺", "少", "拒绝", "无回应", "不回复", "敷衍",
    
    # 其他负面
    "难看", "丑", "臭", "刺鼻", "烂", "坏", "破", "无语", "生气", "糟糕", "劣质",
    "失误", "问题", "缺陷", "瑕疵", "不满", "抱怨", "差评",
    
    # 英文
    "bad", "worst", "terrible", "horrible", "awful", "poor", "disappointing",
    "waste", "cheap", "fake", "scam", "complaint", "refund", "broken", "damaged"
]

# Professional category framework
CATEGORY_FRAMEWORK = {
    "🎨 产品体验": {
        "keywords": [
            "质地", "显色", "持妆", "持久", "掉色", "拔干", "干", "润", "保湿", "滋润",
            "卡纹", "唇纹", "顺滑", "轻薄", "厚重", "氧化", "显气色", "显白", "显黑",
            "服帖", "自然", "细腻", "丝滑", "水润", "清爽", "油腻", "粘", "假白",
            "妆感", "上妆", "晕染", "浮粉", "卡粉", "脱妆", "过敏", "刺激", "闷痘",
            "texture", "lasting", "color", "dry", "smooth", "moisturizing"
        ],
        "description": "产品质地、显色度、持久度、上妆效果等核心使用体验"
    },
    "💰 性价比": {
        "keywords": [
            "价格", "性价比", "值不值", "值", "不值", "贵", "便宜", "溢价", "划算",
            "不划算", "活动", "折扣", "优惠", "实惠", "超值", "物超所值", "坑钱",
            "智商税", "白菜价", "良心价",
            "price", "value", "expensive", "cheap", "worth", "discount"
        ],
        "description": "价格合理性、性价比评估、促销满意度"
    },
    "📦 包装设计": {
        "keywords": [
            "包装", "质感", "高级", "颜值", "好看", "外观", "设计", "精致", "大气",
            "时尚", "奢华", "档次", "简约", "复古", "可爱", "少女", "成熟", "雅致",
            "包装盒", "瓶子", "外壳", "盖子",
            "packaging", "design", "appearance", "aesthetic", "premium", "elegant"
        ],
        "description": "外包装质感、视觉设计、品牌形象"
    },
    "🚚 物流配送": {
        "keywords": [
            "物流", "发货", "到货", "快递", "配送", "包装盒", "快", "慢", "及时",
            "延迟", "破损", "漏液", "丢件", "少件", "缺货", "签收", "收货",
            "delivery", "shipping", "logistics", "fast", "slow", "damaged", "late"
        ],
        "description": "配送速度、包装完整性、物流体验"
    },
    "🛡️ 售后服务": {
        "keywords": [
            "售后", "客服", "退货", "退款", "拒退", "换货", "补偿", "处理", "投诉",
            "态度", "热情", "耐心", "专业", "回复", "解决", "理赔", "维权",
            "service", "support", "refund", "return", "complaint", "response", "staff"
        ],
        "description": "客服响应、退换货政策、问题处理"
    },
    "⚠️ 真伪问题": {
        "keywords": [
            "假货", "真假", "正品", "欺诈", "维权", "举报", "骗", "仿冒", "盗版",
            "三无", "假冒", "验证", "防伪", "授权", "官方", "水货", "山寨",
            "fake", "authentic", "fraud", "counterfeit", "genuine", "trust", "real"
        ],
        "description": "产品真伪、品牌信任度、防伪验证"
    },
    "🔄 竞品对比": {
        "keywords": [
            "不如", "比起", "相比", "对比", "更好", "更差", "差不多", "类似",
            "平替", "替代", "同价位", "同档次", "竞品", "其他品牌", "vs",
            "compare", "versus", "alternative", "better", "worse", "similar"
        ],
        "description": "竞品对比、平替推荐、优劣分析"
    },
}

# Risk assessment thresholds
RISK_LEVELS = {
    "critical": {
        "threshold": 0.4,  # 40%+ negative
        "color": "#c0392b",
        "label": "🔴 严重风险",
        "action": "立即删除/公关处理",
        "timeline": "2小时内"
    },
    "high": {
        "threshold": 0.25,
        "color": "#e74c3c",
        "label": "🟠 高风险",
        "action": "评估删除必要性",
        "timeline": "6小时内"
    },
    "medium": {
        "threshold": 0.15,
        "color": "#f39c12",
        "label": "🟡 中风险",
        "action": "监测并准备回应",
        "timeline": "24小时内"
    },
    "low": {
        "threshold": 0,
        "color": "#2ecc71",
        "label": "🟢 低风险",
        "action": "常规监测",
        "timeline": "定期检查"
    }
}

# ============================================================
# 🔍 ADVANCED KEYWORD EXTRACTION
# ============================================================

def build_dynamic_stopwords(brand_names: List[str], kol_names: List[str] = None) -> Set[str]:
    """Build dynamic stopwords including brands and KOLs"""
    stopwords = CHINESE_STOPWORDS.copy()
    stopwords.update(ENGLISH_STOPWORDS)
    
    # Add brand names
    for brand in brand_names:
        if brand:
            brand_lower = brand.lower().strip()
            stopwords.add(brand_lower)
            stopwords.add(brand.upper())
            stopwords.add(brand.title())
            stopwords.add(f"{brand_lower}家")
            stopwords.add(f"{brand_lower}的")
    
    # Add KOL names
    if kol_names:
        for kol in kol_names:
            if kol:
                kol_lower = kol.lower().strip()
                stopwords.add(kol_lower)
                stopwords.add(f"@{kol_lower}")
    
    return stopwords

def extract_keywords_advanced(text: str, stopwords: Set[str]) -> List[str]:
    """Advanced keyword extraction"""
    text = str(text).strip().lower()
    keywords = []
    
    if JIEBA_AVAILABLE:
        jieba_words = jieba.cut(text, cut_all=False)
        for word in jieba_words:
            word = word.strip()
            if len(word) >= 2 and word not in stopwords:
                if not re.match(r'^\d+$', word):
                    if not re.match(r'^[a-z]$', word):
                        keywords.append(word)
    else:
        chinese = re.findall(r'[\u4e00-\u9fff]{2,6}', text)
        english = re.findall(r'[a-z]{3,20}', text)
        keywords = [w for w in chinese + english if w not in stopwords]
    
    return keywords

def categorize_keyword_smart(keyword: str) -> Tuple[str, str]:
    """Smart categorization"""
    kw_lower = keyword.lower()
    
    for category, info in CATEGORY_FRAMEWORK.items():
        for term in info["keywords"]:
            if term.lower() in kw_lower or kw_lower in term.lower():
                return category, info["description"]
    
    return "📌 其他洞察", "其他消费者关注点"

def extract_top_keywords_enhanced(
    posts: List[str],
    brand_names: List[str],
    kol_names: List[str] = None,
    min_frequency: int = 2,
    top_n: int = 20
) -> List[Dict[str, Any]]:
    """Enhanced keyword extraction"""
    
    if not posts:
        return []
    
    stopwords = build_dynamic_stopwords(brand_names, kol_names)
    keyword_posts_map = defaultdict(set)
    
    for idx, post in enumerate(posts):
        keywords = extract_keywords_advanced(post, stopwords)
        for kw in set(keywords):
            keyword_posts_map[kw].add(idx)
    
    keyword_counts = {kw: len(posts) for kw, posts in keyword_posts_map.items()}
    keyword_counts = {k: v for k, v in keyword_counts.items() if v >= min_frequency}
    
    ranked = sorted(keyword_counts.items(), key=lambda x: -x[1])[:top_n]
    
    results = []
    for keyword, count in ranked:
        category, cat_desc = categorize_keyword_smart(keyword)
        
        keyword_posts_list = [posts[idx] for idx in keyword_posts_map[keyword]]
        sentiment = calculate_sentiment_for_keyword(keyword_posts_list)
        
        # Determine priority
        if count >= 10:
            priority = "High"
            status = "🔴 战略优先"
        elif count >= 5:
            priority = "Medium"
            status = "🟠 验证模式"
        else:
            priority = "Low"
            status = "🟡 新兴信号"
        
        results.append({
            "keyword": keyword,
            "mentions": count,
            "category": category,
            "category_desc": cat_desc,
            "priority": priority,
            "status": status,
            "sentiment_score": sentiment["net_sentiment"],
            "positive_ratio": sentiment["positive_pct"],
            "negative_ratio": sentiment["negative_pct"],
        })
    
    return results

# ============================================================
# 😊 SENTIMENT ANALYSIS
# ============================================================

def analyze_sentiment(text: str) -> str:
    """Rule-based sentiment analysis"""
    text_lower = text.lower()
    
    pos_count = sum(1 for term in POSITIVE_TERMS if term in text_lower)
    neg_count = sum(1 for term in NEGATIVE_TERMS if term in text_lower)
    
    negation_patterns = ["不太", "并不", "不是很", "没那么", "not really", "not very"]
    has_negation = any(pattern in text_lower for pattern in negation_patterns)
    
    if has_negation:
        return "Neutral"
    
    if pos_count > neg_count and pos_count >= 1:
        return "Positive"
    elif neg_count > pos_count and neg_count >= 1:
        return "Negative"
    else:
        return "Neutral"

def calculate_sentiment_distribution(posts: List[str]) -> Dict[str, Any]:
    """Calculate sentiment metrics"""
    if not posts:
        return {
            "positive": 0, "negative": 0, "neutral": 0, "total": 0,
            "positive_pct": 0, "negative_pct": 0, "neutral_pct": 0,
            "net_sentiment": 0, "confidence": "数据不足"
        }
    
    sentiments = [analyze_sentiment(post) for post in posts]
    counter = Counter(sentiments)
    total = len(sentiments)
    
    pos = counter.get("Positive", 0)
    neg = counter.get("Negative", 0)
    neu = counter.get("Neutral", 0)
    
    net_sentiment = (pos - neg) / total if total > 0 else 0
    
    if total < 10:
        confidence = "低 (样本<10)"
    elif total < 30:
        confidence = "中 (样本<30)"
    else:
        confidence = "高 (样本≥30)"
    
    return {
        "positive": pos,
        "negative": neg,
        "neutral": neu,
        "total": total,
        "positive_pct": pos / total if total > 0 else 0,
        "negative_pct": neg / total if total > 0 else 0,
        "neutral_pct": neu / total if total > 0 else 0,
        "net_sentiment": net_sentiment,
        "confidence": confidence
    }

def calculate_sentiment_for_keyword(posts: List[str]) -> Dict[str, Any]:
    """Calculate sentiment for specific posts"""
    return calculate_sentiment_distribution(posts)

# ============================================================
# 🎯 KOL CAMPAIGN MONITORING
# ============================================================

def assess_risk_level(negative_pct: float) -> Dict[str, str]:
    """Assess risk level based on negative percentage"""
    for level_name, level_info in RISK_LEVELS.items():
        if negative_pct >= level_info["threshold"]:
            return {
                "level": level_name,
                "label": level_info["label"],
                "color": level_info["color"],
                "action": level_info["action"],
                "timeline": level_info["timeline"]
            }
    return RISK_LEVELS["low"]

def analyze_kol_performance(
    kol_data: Dict[str, List[str]],
    brand_names: List[str]
) -> List[Dict[str, Any]]:
    """Analyze each KOL's comment performance"""
    
    results = []
    
    for kol_name, comments in kol_data.items():
        if not comments:
            continue
        
        # Sentiment analysis
        sentiment = calculate_sentiment_distribution(comments)
        
        # Risk assessment
        risk = assess_risk_level(sentiment["negative_pct"])
        
        # Keyword extraction
        keywords = extract_top_keywords_enhanced(
            comments, brand_names, [kol_name], min_frequency=1, top_n=10
        )
        
        # Find negative comments
        negative_comments = [
            c for c in comments if analyze_sentiment(c) == "Negative"
        ]
        
        results.append({
            "kol_name": kol_name,
            "total_comments": len(comments),
            "sentiment": sentiment,
            "risk": risk,
            "keywords": keywords,
            "negative_comments": negative_comments,
            "engagement_score": len(comments)  # Simple metric
        })
    
    return results

# ============================================================
# 📊 WORD CLOUD GENERATION
# ============================================================

def create_word_cloud(posts: List[str], brand_names: List[str], kol_names: List[str] = None, title: str = "词云"):
    """Create word cloud"""
    if not WORDCLOUD_AVAILABLE or not posts:
        return None
    
    stopwords = build_dynamic_stopwords(brand_names, kol_names)
    
    all_keywords = []
    for post in posts:
        keywords = extract_keywords_advanced(post, stopwords)
        all_keywords.extend(keywords)
    
    if not all_keywords:
        return None
    
    keyword_freq = Counter(all_keywords)
    
    try:
           # --- Use font inside repo (works on Streamlit Cloud) ---
    base_dir = os.path.dirname(__file__)
    font_path = os.path.join(base_dir, "assets", "fonts", "NotoSansSC-Regular.otf")

    if not os.path.exists(font_path):
        st.warning("Chinese font not found in assets/fonts/. Please upload NotoSansSC-Regular.otf.")
        
        wc = WordCloud(
            width=1200,
            height=600,
            background_color='white',
            font_path=font_path,
            max_words=100,
            relative_scaling=0.5,
            colormap='viridis',
            prefer_horizontal=0.7,
            min_font_size=10,
            max_font_size=100
        ).generate_from_frequencies(keyword_freq)
        
        fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(title, fontsize=16, pad=20, weight='bold')
        
        plt.tight_layout(pad=0)
        
        return fig
        
    except Exception as e:
        st.warning(f"词云生成失败: {str(e)}")
        return None

# ============================================================
# 📈 VISUALIZATIONS
# ============================================================

def create_sentiment_gauge(sentiment_data: Dict, brand_name: str):
    """Sentiment gauge chart"""
    if not PLOTLY_AVAILABLE:
        return None
    
    net_sentiment = sentiment_data.get("net_sentiment", 0)
    gauge_value = (net_sentiment + 1) * 50
    
    if net_sentiment > 0.3:
        color = "#2ecc71"
    elif net_sentiment > 0:
        color = "#f39c12"
    elif net_sentiment > -0.3:
        color = "#e74c3c"
    else:
        color = "#c0392b"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = gauge_value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {
            'text': f"<b>{brand_name}</b><br><span style='font-size:0.8em;color:gray'>净情感指数</span>",
            'font': {'size': 20}
        },
        delta = {'reference': 50, 'increasing': {'color': "green"}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1},
            'bar': {'color': color, 'thickness': 0.3},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 25], 'color': '#ffebee'},
                {'range': [25, 40], 'color': '#fff3e0'},
                {'range': [40, 60], 'color': '#fff9c4'},
                {'range': [60, 75], 'color': '#e8f5e9'},
                {'range': [75, 100], 'color': '#c8e6c9'}
            ],
            'threshold': {
                'line': {'color': "gray", 'width': 2},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        font={'family': "Arial, sans-serif"},
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig

def create_kol_comparison_chart(kol_results: List[Dict]):
    """KOL performance comparison"""
    if not PLOTLY_AVAILABLE or not kol_results:
        return None
    
    kol_names = [r["kol_name"] for r in kol_results]
    engagement = [r["engagement_score"] for r in kol_results]
    negative_pct = [r["sentiment"]["negative_pct"] * 100 for r in kol_results]
    positive_pct = [r["sentiment"]["positive_pct"] * 100 for r in kol_results]
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('评论数量', '情感分布'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}]]
    )
    
    # Engagement
    fig.add_trace(
        go.Bar(x=kol_names, y=engagement, name="评论数", marker_color='#3498db'),
        row=1, col=1
    )
    
    # Sentiment
    fig.add_trace(
        go.Bar(x=kol_names, y=positive_pct, name="正面%", marker_color='#2ecc71'),
        row=1, col=2
    )
    fig.add_trace(
        go.Bar(x=kol_names, y=negative_pct, name="负面%", marker_color='#e74c3c'),
        row=1, col=2
    )
    
    fig.update_layout(
        height=400,
        showlegend=True,
        title_text="<b>KOL表现对比</b>",
        barmode='group'
    )
    
    return fig

def create_risk_radar_chart(kol_results: List[Dict]):
    """Risk assessment radar chart"""
    if not PLOTLY_AVAILABLE or not kol_results:
        return None
    
    fig = go.Figure()
    
    for result in kol_results:
        kol_name = result["kol_name"]
        sentiment = result["sentiment"]
        
        # Calculate risk metrics
        categories = ['负面率', '中性率', '正面率', '互动量', '风险等级']
        
        risk_score_map = {"critical": 100, "high": 75, "medium": 50, "low": 25}
        risk_score = risk_score_map.get(result["risk"]["level"], 0)
        
        values = [
            sentiment["negative_pct"] * 100,
            sentiment["neutral_pct"] * 100,
            sentiment["positive_pct"] * 100,
            min(result["engagement_score"] / 10 * 100, 100),  # Normalize
            risk_score
        ]
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=kol_name
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100])
        ),
        showlegend=True,
        title="<b>KOL风险雷达图</b>",
        height=500
    )
    
    return fig

# ============================================================
# 🎨 MODERN UI STYLING
# ============================================================

def apply_modern_styling():
    """Modern professional UI"""
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        * {
            font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif;
        }
        
        .main {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        }
        
        .main-header {
            font-size: 3rem;
            font-weight: 700;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
            letter-spacing: -1px;
        }
        
        .sub-header {
            font-size: 1.1rem;
            color: #6c757d;
            font-weight: 400;
            margin-bottom: 2rem;
        }
        
        .insight-card {
            background: white;
            border-radius: 16px;
            padding: 1.5rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.07);
            margin: 1rem 0;
            border: 1px solid #e9ecef;
            transition: all 0.3s ease;
        }
        
        .insight-card:hover {
            box-shadow: 0 8px 12px rgba(0,0,0,0.1);
            transform: translateY(-2px);
        }
        
        .explanation-box {
            background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
            border-left: 4px solid #2196f3;
            border-radius: 8px;
            padding: 1rem 1.2rem;
            margin: 1rem 0;
            font-size: 0.9rem;
            line-height: 1.6;
        }
        
        .explanation-title {
            font-weight: 600;
            color: #1976d2;
            margin-bottom: 0.5rem;
            font-size: 1rem;
        }
        
        .metric-container {
            background: white;
            border-radius: 12px;
            padding: 1.2rem;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            border: 1px solid #e9ecef;
        }
        
        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            color: #2c3e50;
            margin: 0.5rem 0;
        }
        
        .metric-label {
            font-size: 0.85rem;
            color: #6c757d;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .risk-critical {
            background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
            border-left: 5px solid #c0392b;
            border-radius: 12px;
            padding: 1.2rem;
            margin: 0.8rem 0;
        }
        
        .risk-high {
            background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
            border-left: 5px solid #e74c3c;
            border-radius: 12px;
            padding: 1.2rem;
            margin: 0.8rem 0;
        }
        
        .risk-medium {
            background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
            border-left: 5px solid #f39c12;
            border-radius: 12px;
            padding: 1.2rem;
            margin: 0.8rem 0;
        }
        
        .risk-low {
            background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
            border-left: 5px solid #2ecc71;
            border-radius: 12px;
            padding: 1.2rem;
            margin: 0.8rem 0;
        }
        
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
            background-color: white;
            padding: 0.8rem;
            border-radius: 12px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 3rem;
            font-size: 0.95rem;
            font-weight: 500;
            border-radius: 8px;
            padding: 0 1.5rem;
            color: #6c757d;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            box-shadow: 0 4px 6px rgba(102, 126, 234, 0.3);
        }
        
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
    </style>
    """, unsafe_allow_html=True)

# ============================================================
# 📄 CAMPAIGN REPORT GENERATION
# ============================================================

def generate_campaign_report_data(
    campaign_name: str,
    kol_results: List[Dict],
    brand_name: str
) -> Dict[str, Any]:
    """Generate comprehensive campaign report data"""
    
    # Overall metrics
    total_comments = sum(r["total_comments"] for r in kol_results)
    total_negative = sum(len(r["negative_comments"]) for r in kol_results)
    
    # Average sentiment
    avg_positive = np.mean([r["sentiment"]["positive_pct"] for r in kol_results])
    avg_negative = np.mean([r["sentiment"]["negative_pct"] for r in kol_results])
    avg_neutral = np.mean([r["sentiment"]["neutral_pct"] for r in kol_results])
    
    # Risk summary
    risk_counts = Counter([r["risk"]["level"] for r in kol_results])
    
    # Top issues
    all_negative_comments = []
    for r in kol_results:
        all_negative_comments.extend(r["negative_comments"])
    
    negative_keywords = extract_top_keywords_enhanced(
        all_negative_comments,
        [brand_name],
        min_frequency=1,
        top_n=10
    ) if all_negative_comments else []
    
    return {
        "campaign_name": campaign_name,
        "brand_name": brand_name,
        "date": date.today().strftime("%Y-%m-%d"),
        "kol_count": len(kol_results),
        "total_comments": total_comments,
        "total_negative": total_negative,
        "avg_positive_pct": avg_positive,
        "avg_negative_pct": avg_negative,
        "avg_neutral_pct": avg_neutral,
        "risk_summary": {
            "critical": risk_counts.get("critical", 0),
            "high": risk_counts.get("high", 0),
            "medium": risk_counts.get("medium", 0),
            "low": risk_counts.get("low", 0)
        },
        "kol_results": kol_results,
        "negative_keywords": negative_keywords
    }

# ============================================================
# 📱 STREAMLIT APP
# ============================================================

st.set_page_config(
    page_title="品牌洞察平台 Ultimate",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_modern_styling()

# Header
st.markdown('<div class="main-header">🎯 品牌洞察平台 Ultimate Edition</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">McKinsey级分析 · KOL监测 · Campaign追踪 · 风险评估 · 智能报告</div>', unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("### ⚙️ 分析配置")
    
    data_source = st.selectbox(
        "数据来源",
        ["🔍 小红书", "🛒 电商评论", "📱 抖音", "💬 微博"]
    )
    
    st.markdown("---")
    st.markdown("### 🏢 品牌设置")
    
    primary_brand = st.text_input("主品牌名称", value="YSL")
    
    enable_competitor = st.toggle("启用竞品分析", value=False)
    
    if enable_competitor:
        competitor_brand = st.text_input("竞品名称", value="Dior")
    else:
        competitor_brand = ""
    
    st.markdown("---")
    st.markdown("### 👥 KOL监测")
    
    enable_kol_monitoring = st.toggle("启用KOL监测", value=False)
    
    if enable_kol_monitoring:
        campaign_name = st.text_input("Campaign名称", value="春季新品推广")
        
        kol_input = st.text_area(
            "输入KOL名称（每行一个）",
            value="李佳琦\n薇娅\n骆王宇",
            height=100
        )
        kol_names_list = [name.strip() for name in kol_input.split('\n') if name.strip()]
    else:
        campaign_name = ""
        kol_names_list = []
    
    st.markdown("---")
    
    with st.expander("🔧 高级设置"):
        min_frequency = st.slider("最小关键词频次", 1, 5, 2)
        top_n = st.slider("展示关键词数量", 10, 30, 20)
        enable_wordcloud = st.checkbox("启用词云", value=True)
        risk_auto_flag = st.checkbox("自动标记高风险", value=True)
    
    st.markdown("---")
    st.markdown("### 📊 系统状态")
    
    if PLOTLY_AVAILABLE:
        st.success("✅ 图表引擎")
    if JIEBA_AVAILABLE:
        st.success("✅ 中文分词")
    if WORDCLOUD_AVAILABLE:
        st.success("✅ 词云生成")

# ============================================================
# MAIN TABS
# ============================================================

if enable_kol_monitoring:
    tabs = st.tabs([
        "📥 数据输入",
        "📊 分析看板",
        "👥 KOL监测",
        "⚠️ 风险评估",
        "📄 Campaign报告"
    ])
    
    tab_input = tabs[0]
    tab_dashboard = tabs[1]
    tab_kol = tabs[2]
    tab_risk = tabs[3]
    tab_campaign = tabs[4]
else:
    tabs = st.tabs([
        "📥 数据输入",
        "📊 分析看板",
        "😊 情感分析",
        "💡 战略洞察",
        "📄 报告导出"
    ])
    
    tab_input = tabs[0]
    tab_dashboard = tabs[1]
    tab_sentiment = tabs[2]
    tab_insights = tabs[3]
    tab_report = tabs[4]

# Initialize session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

if 'kol_analysis' not in st.session_state:
    st.session_state.kol_analysis = None

# ============================================================
# TAB 1: DATA INPUT
# ============================================================

with tab_input:
    st.markdown("## 📥 数据输入")
    
    st.markdown("""
    <div class="explanation-box">
        <div class="explanation-title">📖 使用说明</div>
        <strong>常规分析：</strong>上传品牌评论数据进行情感分析和关键词提取<br>
        <strong>KOL监测：</strong>启用后，可按KOL分别上传评论，进行风险评估和Campaign追踪<br>
        <strong>建议样本量：</strong>每个KOL 20+条评论，整体30+条以获得准确洞察
    </div>
    """, unsafe_allow_html=True)
    
    def load_data_simple(label: str, key: str) -> List[str]:
        """Simple data loading"""
        st.markdown(f"### {label}")
        
        uploaded_file = st.file_uploader(f"上传CSV", type=["csv"], key=f"{key}_file")
        
        posts = []
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                
                text_col = st.selectbox(f"选择文本列", df.columns.tolist(), key=f"{key}_col")
                
                raw_posts = df[text_col].dropna().astype(str).tolist()
                
                seen = set()
                for post in raw_posts:
                    post = post.strip()
                    if len(post) >= 5 and post not in seen:
                        posts.append(post)
                        seen.add(post)
                
                st.success(f"✅ 已加载 {len(posts)} 条评论")
                
            except Exception as e:
                st.error(f"❌ 加载失败: {str(e)}")
        
        manual_input = st.text_area(
            f"或手动粘贴评论（每行一条）",
            height=150,
            key=f"{key}_manual"
        )
        
        if manual_input.strip() and not posts:
            lines = manual_input.strip().split('\n')
            seen = set()
            for line in lines:
                line = line.strip()
                if len(line) >= 5 and line not in seen:
                    posts.append(line)
                    seen.add(line)
            
            if posts:
                st.success(f"✅ 已加载 {len(posts)} 条评论")
        
        return posts
    
    if enable_kol_monitoring:
        st.markdown("### 👥 按KOL分别上传数据")
        
        kol_data_dict = {}
        
        cols = st.columns(min(len(kol_names_list), 3))
        
        for idx, kol_name in enumerate(kol_names_list):
            with cols[idx % 3]:
                posts = load_data_simple(f"📱 {kol_name}", f"kol_{idx}")
                if posts:
                    kol_data_dict[kol_name] = posts
        
        if st.button("💾 保存KOL数据", type="primary"):
            st.session_state.kol_data = kol_data_dict
            st.success(f"✅ 已保存 {len(kol_data_dict)} 个KOL的数据")
    
    else:
        # Regular data input
        col_brand_1, col_brand_2 = st.columns(2)
        
        with col_brand_1:
            primary_posts = load_data_simple(f"🎯 {primary_brand}", "primary")
        
        with col_brand_2:
            if enable_competitor:
                competitor_posts = load_data_simple(f"🔄 {competitor_brand}", "competitor")
            else:
                competitor_posts = []

# ============================================================
# TAB 2: ANALYSIS DASHBOARD
# ============================================================

with tab_dashboard:
    st.markdown("## 📊 综合分析看板")
    
    if st.button("🚀 开始智能分析", type="primary", use_container_width=True):
        
        if enable_kol_monitoring:
            if not hasattr(st.session_state, 'kol_data') or not st.session_state.kol_data:
                st.error("⚠️ 请先在'数据输入'标签页上传KOL数据")
                st.stop()
            
            with st.spinner("🔄 分析KOL数据中..."):
                kol_results = analyze_kol_performance(
                    st.session_state.kol_data,
                    [primary_brand]
                )
                
                st.session_state.kol_analysis = {
                    "campaign_name": campaign_name,
                    "brand_name": primary_brand,
                    "kol_results": kol_results,
                    "brand_names": [primary_brand]
                }
                
                st.success("✅ KOL分析完成！")
        
        else:
            if 'primary_posts' not in locals() or not primary_posts:
                st.error("⚠️ 请先在'数据输入'标签页提供数据")
                st.stop()
            
            with st.spinner("🔄 分析中..."):
                brand_names = [primary_brand]
                if enable_competitor and competitor_brand:
                    brand_names.append(competitor_brand)
                
                primary_keywords = extract_top_keywords_enhanced(
                    primary_posts, brand_names, min_frequency=min_frequency, top_n=top_n
                )
                primary_sentiment = calculate_sentiment_distribution(primary_posts)
                
                if enable_competitor and 'competitor_posts' in locals() and competitor_posts:
                    competitor_keywords = extract_top_keywords_enhanced(
                        competitor_posts, brand_names, min_frequency=min_frequency, top_n=top_n
                    )
                    competitor_sentiment = calculate_sentiment_distribution(competitor_posts)
                else:
                    competitor_keywords = []
                    competitor_sentiment = {}
                
                st.session_state.analysis_results = {
                    "primary": {
                        "brand": primary_brand,
                        "posts": primary_posts,
                        "keywords": primary_keywords,
                        "sentiment": primary_sentiment,
                    },
                    "competitor": {
                        "brand": competitor_brand,
                        "posts": competitor_posts,
                        "keywords": competitor_keywords,
                        "sentiment": competitor_sentiment,
                    } if enable_competitor and competitor_posts else None,
                    "brand_names": brand_names
                }
                
                st.success("✅ 分析完成！")
    
    # Display results
    if enable_kol_monitoring and st.session_state.kol_analysis:
        kol_analysis = st.session_state.kol_analysis
        kol_results = kol_analysis["kol_results"]
        
        st.markdown("---")
        st.markdown("### 📈 Campaign总览")
        
        col1, col2, col3, col4 = st.columns(4)
        
        total_comments = sum(r["total_comments"] for r in kol_results)
        total_negative = sum(len(r["negative_comments"]) for r in kol_results)
        avg_negative_pct = np.mean([r["sentiment"]["negative_pct"] for r in kol_results])
        high_risk_count = sum(1 for r in kol_results if r["risk"]["level"] in ["critical", "high"])
        
        with col1:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">总评论数</div>
                <div class="metric-value">{total_comments}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">负面评论</div>
                <div class="metric-value" style="color:#e74c3c;">{total_negative}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">平均负面率</div>
                <div class="metric-value" style="color:{'#e74c3c' if avg_negative_pct > 0.25 else '#f39c12'};">{avg_negative_pct:.1%}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">高风险KOL</div>
                <div class="metric-value" style="color:{'#c0392b' if high_risk_count > 0 else '#2ecc71'};">{high_risk_count}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # KOL comparison charts
        if PLOTLY_AVAILABLE:
            st.markdown("### 📊 KOL表现对比")
            
            st.markdown("""
            <div class="explanation-box">
                <div class="explanation-title">📖 图表说明</div>
                <strong>左图：</strong>各KOL的评论互动量<br>
                <strong>右图：</strong>各KOL的正面/负面评论占比<br>
                <strong>雷达图：</strong>多维度风险评估（越靠外圈风险越高）
            </div>
            """, unsafe_allow_html=True)
            
            fig_comparison = create_kol_comparison_chart(kol_results)
            if fig_comparison:
                st.plotly_chart(fig_comparison, use_container_width=True)
            
            fig_radar = create_risk_radar_chart(kol_results)
            if fig_radar:
                st.plotly_chart(fig_radar, use_container_width=True)
        
        # Word clouds
        if enable_wordcloud and WORDCLOUD_AVAILABLE:
            st.markdown("---")
            st.markdown("### ☁️ KOL词云对比")
            
            st.markdown("""
            <div class="explanation-box">
                <div class="explanation-title">📖 词云说明</div>
                展示各KOL评论区的高频关键词，字体大小代表提及频率。已自动过滤KOL名称和无意义词。
            </div>
            """, unsafe_allow_html=True)
            
            cols_wc = st.columns(min(len(kol_results), 3))
            
            for idx, result in enumerate(kol_results):
                with cols_wc[idx % 3]:
                    st.markdown(f"#### {result['kol_name']}")
                    
                    if result["total_comments"] > 0:
                        all_comments = st.session_state.kol_data.get(result["kol_name"], [])
                        wc_fig = create_word_cloud(
                            all_comments,
                            [primary_brand],
                            [result["kol_name"]],
                            f"{result['kol_name']} 评论词云"
                        )
                        if wc_fig:
                            st.pyplot(wc_fig)
                            plt.close()
    
    elif st.session_state.analysis_results:
        # Regular analysis display
        results = st.session_state.analysis_results
        primary_data = results["primary"]
        
        st.markdown("---")
        st.markdown("### 📈 快速概览")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">样本量</div>
                <div class="metric-value">{len(primary_data["posts"])}</div>
                <div style="font-size:0.8rem;color:#6c757d;">{primary_data["sentiment"]["confidence"]}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            net_sent = primary_data["sentiment"]["net_sentiment"]
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">净情感</div>
                <div class="metric-value" style="color:{'#2ecc71' if net_sent > 0.2 else '#e74c3c' if net_sent < -0.2 else '#f39c12'};">{net_sent:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            high_priority = sum(1 for kw in primary_data["keywords"] if kw["priority"] == "High")
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">高优先级</div>
                <div class="metric-value" style="color:{'#e74c3c' if high_priority > 0 else '#2ecc71'};">{high_priority}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">关键词</div>
                <div class="metric-value">{len(primary_data["keywords"])}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Visualizations
        if PLOTLY_AVAILABLE:
            st.markdown("### 📊 可视化洞察")
            
            st.markdown("""
            <div class="explanation-box">
                <div class="explanation-title">📖 图表说明</div>
                <strong>情感仪表盘：</strong>整体情感倾向，50为中性，越高越正面<br>
                <strong>词云：</strong>字体大小=提及频率，颜色仅用于区分
            </div>
            """, unsafe_allow_html=True)
            
            fig_gauge = create_sentiment_gauge(primary_data["sentiment"], primary_brand)
            if fig_gauge:
                st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Word clouds
        if enable_wordcloud and WORDCLOUD_AVAILABLE:
            st.markdown("---")
            st.markdown("### ☁️ 关键词词云")
            
            wc_fig = create_word_cloud(
                primary_data["posts"],
                results["brand_names"],
                title=f"{primary_brand} 高频关键词"
            )
            if wc_fig:
                st.pyplot(wc_fig)
                plt.close()

# ============================================================
# TAB 3: KOL MONITORING (if enabled) or SENTIMENT ANALYSIS
# ============================================================

if enable_kol_monitoring:
    with tab_kol:
        st.markdown("## 👥 KOL详细监测")
        
        if not st.session_state.kol_analysis:
            st.info("👈 请先在'分析看板'运行分析")
            st.stop()
        
        kol_analysis = st.session_state.kol_analysis
        kol_results = kol_analysis["kol_results"]
        
        st.markdown("""
        <div class="explanation-box">
            <div class="explanation-title">📖 KOL监测说明</div>
            针对每个KOL的评论区进行独立分析，评估互动质量和潜在风险。<br>
            <strong>绿色</strong>=低风险，<strong>黄色</strong>=中风险，<strong>橙色</strong>=高风险，<strong>红色</strong>=严重风险
        </div>
        """, unsafe_allow_html=True)
        
        for result in kol_results:
            kol_name = result["kol_name"]
            sentiment = result["sentiment"]
            risk = result["risk"]
            
            risk_class = f"risk-{risk['level']}"
            
            with st.container():
                st.markdown(f"""
                <div class="{risk_class}">
                    <h3>📱 {kol_name}</h3>
                    <strong>风险等级：</strong>{risk['label']}<br>
                    <strong>评论数：</strong>{result['total_comments']} 条<br>
                    <strong>情感分布：</strong>正面 {sentiment['positive_pct']:.1%} | 中性 {sentiment['neutral_pct']:.1%} | 负面 {sentiment['negative_pct']:.1%}<br>
                    <strong>建议行动：</strong>{risk['action']}<br>
                    <strong>处理时限：</strong>{risk['timeline']}
                </div>
                """, unsafe_allow_html=True)
                
                if result["negative_comments"]:
                    with st.expander(f"⚠️ 查看负面评论 ({len(result['negative_comments'])}条)"):
                        for idx, comment in enumerate(result["negative_comments"][:10], 1):
                            st.markdown(f"{idx}. {comment}")
                        
                        if len(result["negative_comments"]) > 10:
                            st.info(f"还有 {len(result['negative_comments']) - 10} 条负面评论未显示")
                
                if result["keywords"]:
                    with st.expander(f"🔑 关键词分析 (Top {len(result['keywords'])})"):
                        df_kw = pd.DataFrame(result["keywords"])
                        st.dataframe(
                            df_kw[["keyword", "mentions", "category", "sentiment_score"]],
                            use_container_width=True,
                            hide_index=True
                        )
                
                st.markdown("---")

else:
    with tab_sentiment:
        st.markdown("## 😊 情感分析")
        
        if not st.session_state.analysis_results:
            st.info("👈 请先在'分析看板'运行分析")
            st.stop()
        
        results = st.session_state.analysis_results
        primary_data = results["primary"]
        
        st.markdown("""
        <div class="explanation-box">
            <div class="explanation-title">📖 情感分析说明</div>
            基于正负面关键词规则判断每条评论情感倾向。<br>
            <strong>净情感分数 = (正面 - 负面) / 总数</strong>，范围 -1 到 +1
        </div>
        """, unsafe_allow_html=True)
        
        sent_data = primary_data["sentiment"]
        
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("😊 正面", f"{sent_data['positive_pct']:.1%}", delta=f"{sent_data['positive']}条")
        col_b.metric("😐 中性", f"{sent_data['neutral_pct']:.1%}", delta=f"{sent_data['neutral']}条")
        col_c.metric("😞 负面", f"{sent_data['negative_pct']:.1%}", delta=f"{sent_data['negative']}条", delta_color="inverse")

# ============================================================
# TAB 4: RISK ASSESSMENT or STRATEGIC INSIGHTS
# ============================================================

if enable_kol_monitoring:
    with tab_risk:
        st.markdown("## ⚠️ 风险评估与应对建议")
        
        if not st.session_state.kol_analysis:
            st.info("👈 请先在'分析看板'运行分析")
            st.stop()
        
        kol_analysis = st.session_state.kol_analysis
        kol_results = kol_analysis["kol_results"]
        
        st.markdown("""
        <div class="explanation-box">
            <div class="explanation-title">📖 风险评估标准</div>
            <strong>🔴 严重风险(≥40%负面)：</strong>立即删除负面评论或启动公关应对<br>
            <strong>🟠 高风险(25-40%负面)：</strong>评估删除必要性，准备回应话术<br>
            <strong>🟡 中风险(15-25%负面)：</strong>密切监测，准备应对预案<br>
            <strong>🟢 低风险(<15%负面)：</strong>常规监测即可
        </div>
        """, unsafe_allow_html=True)
        
        # High risk KOLs
        high_risk_kols = [r for r in kol_results if r["risk"]["level"] in ["critical", "high"]]
        
        if high_risk_kols:
            st.markdown("### 🚨 需要立即处理的KOL")
            
            for result in high_risk_kols:
                risk_class = f"risk-{result['risk']['level']}"
                
                st.markdown(f"""
                <div class="{risk_class}">
                    <h4>{result['risk']['label']} {result['kol_name']}</h4>
                    <strong>负面率：</strong>{result['sentiment']['negative_pct']:.1%} ({len(result['negative_comments'])}条)<br>
                    <strong>建议行动：</strong><br>
                    1. {result['risk']['action']}<br>
                    2. 联系KOL沟通删除事宜<br>
                    3. 准备官方回应话术<br>
                    4. 监测后续舆情变化<br>
                    <strong>处理时限：</strong><span style="color:red;font-weight:bold;">{result['risk']['timeline']}</span>
                </div>
                """, unsafe_allow_html=True)
                
                # Action plan
                with st.expander("📋 详细应对方案"):
                    st.markdown(f"""
                    **Step 1: 评估删除必要性**
                    - 负面评论数量: {len(result['negative_comments'])}条
                    - 是否涉及产品质量问题: 需人工判断
                    - 是否涉及虚假宣传: 需人工判断
                    
                    **Step 2: 联系KOL**
                    - 发送删除请求邮件
                    - 说明理由和影响
                    - 提供补偿方案（如需要）
                    
                    **Step 3: 公关应对**
                    - 准备官方声明
                    - 在其他平台发布正面内容
                    - 监测品牌舆情变化
                    
                    **Step 4: 持续追踪**
                    - 每2小时检查一次
                    - 记录处理进展
                    - 评估效果
                    """)
                
                st.markdown("---")
        
        else:
            st.success("✅ 当前无高风险KOL，继续保持监测")
        
        # All KOLs summary
        st.markdown("### 📊 全部KOL风险总览")
        
        risk_summary_data = []
        for result in kol_results:
            risk_summary_data.append({
                "KOL": result["kol_name"],
                "评论数": result["total_comments"],
                "负面数": len(result["negative_comments"]),
                "负面率": f"{result['sentiment']['negative_pct']:.1%}",
                "风险等级": result["risk"]["label"],
                "建议行动": result["risk"]["action"]
            })
        
        df_risk = pd.DataFrame(risk_summary_data)
        st.dataframe(df_risk, use_container_width=True, hide_index=True)

else:
    with tab_insights:
        st.markdown("## 💡 战略洞察")
        
        if not st.session_state.analysis_results:
            st.info("👈 请先在'分析看板'运行分析")
            st.stop()
        
        results = st.session_state.analysis_results
        primary_data = results["primary"]
        
        st.markdown("""
        <div class="explanation-box">
            <div class="explanation-title">📖 战略建议说明</div>
            基于关键词频次和情感自动生成优先级建议。<br>
            <strong>高优先级(≥10次)：</strong>24-48h处理<br>
            <strong>中优先级(5-9次)：</strong>1周内处理<br>
            <strong>低优先级(2-4次)：</strong>2周内监测
        </div>
        """, unsafe_allow_html=True)
        
        high_priority = [kw for kw in primary_data["keywords"] if kw["priority"] == "High"]
        
        if high_priority:
            st.markdown("### 🔴 高优先级")
            for kw in high_priority:
                st.markdown(f"""
                <div class="risk-high">
                    <h4>{kw['keyword']} ({kw['category']})</h4>
                    <strong>提及：</strong>{kw['mentions']}次<br>
                    <strong>情感：</strong>{kw['sentiment_score']:.2f}<br>
                    <strong>行动：</strong>立即处理相关问题
                </div>
                """, unsafe_allow_html=True)

# ============================================================
# TAB 5: CAMPAIGN REPORT or EXPORT
# ============================================================

if enable_kol_monitoring:
    with tab_campaign:
        st.markdown("## 📄 Campaign分析报告")
        
        if not st.session_state.kol_analysis:
            st.info("👈 请先在'分析看板'运行分析")
            st.stop()
        
        kol_analysis = st.session_state.kol_analysis
        
        st.markdown("""
        <div class="explanation-box">
            <div class="explanation-title">📖 报告说明</div>
            自动生成Campaign执行报告，包含KOL表现、风险评估、负面评论汇总等内容。<br>
            可导出CSV/JSON格式，或生成Word文档报告（需要安装docx相关依赖）。
        </div>
        """, unsafe_allow_html=True)
        
        # Generate report data
        report_data = generate_campaign_report_data(
            campaign_name,
            kol_analysis["kol_results"],
            kol_analysis["brand_name"]
        )
        
        # Display summary
        st.markdown("### 📊 Campaign执行摘要")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("参与KOL数", report_data["kol_count"])
            st.metric("总评论数", report_data["total_comments"])
        
        with col2:
            st.metric("平均正面率", f"{report_data['avg_positive_pct']:.1%}")
            st.metric("平均负面率", f"{report_data['avg_negative_pct']:.1%}")
        
        with col3:
            st.metric("严重风险", report_data["risk_summary"]["critical"], delta_color="inverse")
            st.metric("高风险", report_data["risk_summary"]["high"], delta_color="inverse")
        
        st.markdown("---")
        
        # Top negative keywords
        if report_data["negative_keywords"]:
            st.markdown("### ⚠️ 主要负面关键词")
            
            df_neg_kw = pd.DataFrame(report_data["negative_keywords"])
            st.dataframe(
                df_neg_kw[["keyword", "mentions", "category", "sentiment_score"]].head(10),
                use_container_width=True,
                hide_index=True
            )
        
        st.markdown("---")
        
        # Export options
        st.markdown("### 💾 导出报告")
        
        col_dl1, col_dl2, col_dl3 = st.columns(3)
        
        # Prepare export data
        kol_summary_data = []
        for r in report_data["kol_results"]:
            kol_summary_data.append({
                "KOL": r["kol_name"],
                "评论数": r["total_comments"],
                "正面率": f"{r['sentiment']['positive_pct']:.1%}",
                "负面率": f"{r['sentiment']['negative_pct']:.1%}",
                "风险等级": r["risk"]["label"],
                "负面评论数": len(r["negative_comments"])
            })
        
        df_kol_summary = pd.DataFrame(kol_summary_data)
        
        with col_dl1:
            st.download_button(
                label="📥 KOL汇总CSV",
                data=df_kol_summary.to_csv(index=False).encode('utf-8-sig'),
                file_name=f"{campaign_name}_KOL汇总_{date.today()}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col_dl2:
            json_export = json.dumps(report_data, ensure_ascii=False, indent=2, default=str)
            st.download_button(
                label="📥 完整报告JSON",
                data=json_export.encode('utf-8'),
                file_name=f"{campaign_name}_完整报告_{date.today()}.json",
                mime="application/json",
                use_container_width=True
            )
        
        with col_dl3:
            st.info("Word文档生成功能开发中")
        
        # Preview
        with st.expander("📊 KOL汇总预览"):
            st.dataframe(df_kol_summary, use_container_width=True, hide_index=True)

else:
    with tab_report:
        st.markdown("## 📄 报告导出")
        
        if not st.session_state.analysis_results:
            st.info("👈 请先在'分析看板'运行分析")
            st.stop()
        
        results = st.session_state.analysis_results
        primary_data = results["primary"]
        
        st.markdown("""
        <div class="explanation-box">
            <div class="explanation-title">📖 导出说明</div>
            提供CSV和JSON格式导出，便于后续分析或汇报使用。
        </div>
        """, unsafe_allow_html=True)
        
        keywords_df = pd.DataFrame(primary_data["keywords"])
        
        col_dl1, col_dl2 = st.columns(2)
        
        with col_dl1:
            st.download_button(
                label="📥 下载关键词CSV",
                data=keywords_df.to_csv(index=False).encode('utf-8-sig'),
                file_name=f"{primary_brand}_关键词_{date.today()}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col_dl2:
            json_export = json.dumps(results, ensure_ascii=False, indent=2, default=str)
            st.download_button(
                label="📥 下载完整JSON",
                data=json_export.encode('utf-8'),
                file_name=f"{primary_brand}_分析_{date.today()}.json",
                mime="application/json",
                use_container_width=True
            )

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

col_footer1, col_footer2, col_footer3 = st.columns(3)

with col_footer1:
    st.markdown("**🎯 核心功能**")
    st.markdown("""
    - 精准关键词提取
    - 智能情感分析
    - KOL监测追踪
    - 风险评估预警
    - Campaign报告
    """)

with col_footer2:
    st.markdown("**📊 分析维度**")
    st.markdown("""
    - 产品体验
    - 性价比
    - 包装设计
    - 售后服务
    - 真伪问题
    - 物流配送
    - 竞品对比
    """)

with col_footer3:
    st.markdown("**💡 使用建议**")
    st.markdown("""
    - 每个KOL 20+评论
    - 每天更新监测
    - 关注高风险KOL
    - 及时处理负面
    """)

st.markdown("---")
st.caption("**Ultimate Brand Intelligence Platform** | KOL Monitoring · Risk Assessment · Campaign Tracking")
