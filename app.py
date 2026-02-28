import streamlit as st
import os
import re
import json
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Any, Set
from io import BytesIO
from datetime import date, datetime
from groq import Groq

# ============================================================
# 🎯 CONSULTING-GRADE BRAND INTELLIGENCE PLATFORM
# Professional UI · Complete Analytics · Strategic Insights
# ============================================================

# Optional dependencies
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.error("⚠️ Plotly未安装。请运行: pip install plotly")

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
    """Initialize Groq client with secure key management"""
    api_key = get_groq_api_key()
    
    if not api_key:
        st.sidebar.warning("⚠️ GROQ API KEY 未配置")
        st.sidebar.markdown("""
        **配置方法:**
        
        **方法1: Streamlit Secrets**
        ```bash
        mkdir -p ~/.streamlit
        echo 'GROQ_API_KEY = "your-key"' > ~/.streamlit/secrets.toml
        ```
        
        **方法2: 环境变量**
        ```bash
        export GROQ_API_KEY="your-key"
        ```
        """)
        
        temp_key = st.sidebar.text_input("临时输入 API Key", type="password", key="temp_api_key")
        if temp_key:
            api_key = temp_key
        else:
            return None
    
    try:
        client = Groq(api_key=api_key, timeout=30.0, max_retries=3)
        return client
    except Exception as e:
        st.sidebar.error(f"❌ API初始化失败: {str(e)}")
        return None

client = initialize_groq_client()

# ============================================================
# 📊 ENHANCED CONFIGURATION
# ============================================================

# Comprehensive stopwords (200+ words)
CHINESE_STOPWORDS = {
    # 通用停用词
    "的", "了", "是", "在", "我", "有", "和", "就", "不", "人", "都", "一", "个",
    "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好",
    "自己", "这", "那", "里", "来", "为", "但", "而", "与", "或", "啊", "呀", "吗",
    "呢", "吧", "哦", "哈", "嗯", "唉", "哎", "啦", "么", "嘛", "呗", "得", "吧",
    
    # 无意义高频词
    "真的", "感觉", "觉得", "还是", "然后", "所以", "因为", "如果", "这个", "那个",
    "什么", "怎么", "非常", "比较", "可能", "应该", "肯定", "一定", "绝对", "真是",
    "确实", "简直", "完全", "十分", "特别", "尤其", "总之", "反正", "其实", "本来",
    "已经", "还有", "而且", "不过", "只是", "可是", "但是", "虽然", "尽管", "还好",
    
    # 品牌相关（动态添加）
    "品牌", "牌子", "产品", "东西", "这款", "那款", "这个牌子", "那个牌子",
    "这家", "那家", "商家", "店家",
    
    # 电商/社交平台
    "旗舰店", "官方", "店铺", "卖家", "买家", "客服", "购买", "下单", "收货",
    "小红书", "笔记", "种草", "拔草", "安利", "分享", "推荐", "测评", "开箱",
    
    # 代词和连词
    "他", "她", "它", "们", "我们", "你们", "他们", "她们", "咱们",
}

ENGLISH_STOPWORDS = {
    "the", "is", "are", "was", "were", "be", "been", "being", "a", "an", "and",
    "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "from", "as",
    "it", "this", "that", "these", "those", "i", "you", "he", "she", "we", "they"
}

# Enhanced sentiment terms
POSITIVE_TERMS = [
    # 产品体验
    "好", "棒", "喜欢", "爱", "推荐", "值得", "满意", "完美", "赞", "优秀", "不错",
    "好用", "实用", "舒服", "温和", "柔滑", "显白", "持久", "滋润", "保湿", "清爽",
    "轻薄", "服帖", "自然", "显色", "正", "顺滑", "细腻", "丝滑", "水润", "嫩",
    
    # 价值
    "划算", "超值", "惊喜", "物超所值", "性价比高", "实惠", "便宜", "优惠",
    
    # 外观
    "高级", "精致", "漂亮", "美", "好看", "颜值高", "大气", "时尚", "奢华",
    
    # 英文
    "good", "great", "love", "excellent", "perfect", "amazing", "best", "recommend"
]

NEGATIVE_TERMS = [
    # 产品问题
    "差", "不好", "失望", "后悔", "避雷", "翻车", "坑", "垃圾", "难用", "鸡肋",
    "拔干", "卡纹", "掉色", "氧化", "暗沉", "显黑", "厚重", "油腻", "粘", "假白",
    "不持久", "易掉", "易花", "斑驳", "不均匀", "结块", "起皮", "过敏", "刺激",
    
    # 价格
    "不值", "贵", "太贵", "溢价", "不划算", "性价比低", "坑钱", "智商税",
    
    # 真伪
    "假", "假货", "骗", "欺诈", "仿冒", "不是正品", "有问题", "怀疑",
    
    # 服务
    "退", "退货", "退款", "投诉", "维权", "拒退", "态度差", "慢", "破损",
    
    # 英文
    "bad", "worst", "terrible", "horrible", "awful", "poor", "fake", "scam"
]

# Professional category framework
CATEGORY_FRAMEWORK = {
    "🎨 产品体验": {
        "keywords": [
            "质地", "显色", "持妆", "持久", "掉色", "拔干", "干", "润", "保湿", "滋润",
            "卡纹", "唇纹", "顺滑", "轻薄", "厚重", "氧化", "显气色", "显白", "显黑",
            "服帖", "自然", "细腻", "丝滑", "水润", "清爽", "油腻", "粘", "假白",
            "妆感", "上妆", "晕染", "浮粉", "卡粉", "脱妆", "过敏", "刺激", "闷痘"
        ]
    },
    "💰 性价比": {
        "keywords": [
            "价格", "性价比", "值不值", "值", "不值", "贵", "便宜", "溢价", "划算",
            "不划算", "活动", "折扣", "优惠", "实惠", "超值", "物超所值", "坑钱", "智商税"
        ]
    },
    "📦 包装设计": {
        "keywords": [
            "包装", "质感", "高级", "颜值", "好看", "外观", "设计", "精致", "大气",
            "时尚", "奢华", "档次", "简约", "复古", "可爱", "包装盒", "瓶子"
        ]
    },
    "🚚 物流配送": {
        "keywords": [
            "物流", "发货", "到货", "快递", "配送", "快", "慢", "及时", "延迟",
            "破损", "漏液", "丢件", "少件", "缺货"
        ]
    },
    "🛡️ 售后服务": {
        "keywords": [
            "售后", "客服", "退货", "退款", "拒退", "换货", "补偿", "处理", "投诉",
            "态度", "热情", "耐心", "专业", "回复", "解决"
        ]
    },
    "⚠️ 真伪问题": {
        "keywords": [
            "假货", "真假", "正品", "欺诈", "维权", "举报", "骗", "仿冒", "盗版",
            "三无", "假冒", "验证", "防伪", "授权", "官方"
        ]
    },
    "🔄 竞品对比": {
        "keywords": [
            "不如", "比起", "相比", "对比", "更好", "更差", "差不多", "类似",
            "平替", "替代", "同价位", "同档次", "竞品", "其他品牌"
        ]
    }
}

# ============================================================
# 🎨 CISION-STYLE PROFESSIONAL UI
# ============================================================

def apply_cision_ui():
    """Apply Cision-inspired professional consulting UI"""
    st.markdown("""
    <style>
        /* Import professional fonts */
        @import url('https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@300;400;600;700&family=Inter:wght@400;500;600;700&display=swap');
        
        /* Global reset */
        * {
            font-family: 'Source Sans Pro', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        /* Main container */
        .main {
            background: #f8f9fa;
            padding: 0;
        }
        
        .block-container {
            padding: 2rem 3rem;
            max-width: 1400px;
        }
        
        /* Header */
        .main-header {
            font-size: 2.5rem;
            font-weight: 700;
            color: #1a1a1a;
            margin-bottom: 0.5rem;
            letter-spacing: -0.5px;
        }
        
        .sub-header {
            font-size: 1rem;
            color: #6c757d;
            font-weight: 400;
            margin-bottom: 2rem;
            letter-spacing: 0.3px;
        }
        
        /* Cards - Cision style */
        .metric-card {
            background: white;
            border-radius: 8px;
            padding: 1.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
            border: 1px solid #e9ecef;
            margin-bottom: 1rem;
            transition: all 0.2s ease;
        }
        
        .metric-card:hover {
            box-shadow: 0 4px 8px rgba(0,0,0,0.12);
        }
        
        .metric-value {
            font-size: 2.5rem;
            font-weight: 700;
            color: #2c3e50;
            line-height: 1;
            margin: 0.5rem 0;
        }
        
        .metric-label {
            font-size: 0.875rem;
            color: #6c757d;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-weight: 600;
        }
        
        .metric-delta {
            font-size: 0.875rem;
            margin-top: 0.5rem;
            font-weight: 500;
        }
        
        /* Section headers */
        .section-header {
            font-size: 1.5rem;
            font-weight: 600;
            color: #1a1a1a;
            margin: 2rem 0 1rem 0;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #e9ecef;
        }
        
        /* Info boxes */
        .info-box {
            background: #f8f9fa;
            border-left: 4px solid #007bff;
            border-radius: 4px;
            padding: 1rem 1.2rem;
            margin: 1rem 0;
            font-size: 0.9rem;
            line-height: 1.6;
        }
        
        .info-box-title {
            font-weight: 600;
            color: #007bff;
            margin-bottom: 0.5rem;
        }
        
        /* Tabs - Cision style */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0;
            background-color: white;
            padding: 0;
            border-radius: 8px 8px 0 0;
            border-bottom: 2px solid #e9ecef;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 3.5rem;
            font-size: 0.9rem;
            font-weight: 600;
            border-radius: 0;
            padding: 0 2rem;
            color: #6c757d;
            border-bottom: 3px solid transparent;
            transition: all 0.2s ease;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            background-color: #f8f9fa;
            color: #1a1a1a;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: white;
            color: #007bff;
            border-bottom: 3px solid #007bff;
        }
        
        /* Buttons */
        .stButton > button {
            background: #007bff;
            color: white;
            border: none;
            border-radius: 6px;
            padding: 0.6rem 1.5rem;
            font-weight: 600;
            font-size: 0.9rem;
            letter-spacing: 0.3px;
            transition: all 0.2s ease;
        }
        
        .stButton > button:hover {
            background: #0056b3;
            box-shadow: 0 4px 8px rgba(0,123,255,0.3);
        }
        
        /* DataFrames */
        .dataframe {
            font-size: 0.875rem;
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid #e9ecef;
        }
        
        .dataframe thead th {
            background-color: #f8f9fa;
            color: #1a1a1a;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.75rem;
            letter-spacing: 0.5px;
            padding: 1rem 0.75rem;
        }
        
        .dataframe tbody td {
            padding: 0.75rem;
            border-bottom: 1px solid #f1f3f5;
        }
        
        /* Sidebar */
        .css-1d391kg, [data-testid="stSidebar"] {
            background: white;
            border-right: 1px solid #e9ecef;
        }
        
        [data-testid="stSidebar"] .element-container {
            padding: 0.5rem 0;
        }
        
        /* Metrics grid */
        .metrics-row {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin: 1rem 0;
        }
        
        /* Priority badges */
        .badge {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.3px;
            text-transform: uppercase;
        }
        
        .badge-high {
            background: #fee;
            color: #c00;
        }
        
        .badge-medium {
            background: #ffeaa7;
            color: #d63031;
        }
        
        .badge-low {
            background: #dfe6e9;
            color: #636e72;
        }
        
        /* Chart containers */
        .chart-container {
            background: white;
            border-radius: 8px;
            padding: 1.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
            border: 1px solid #e9ecef;
            margin: 1rem 0;
        }
        
        .chart-title {
            font-size: 1.1rem;
            font-weight: 600;
            color: #1a1a1a;
            margin-bottom: 1rem;
        }
        
        /* Remove default streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Responsive */
        @media (max-width: 768px) {
            .block-container {
                padding: 1rem;
            }
            
            .metric-value {
                font-size: 2rem;
            }
        }
    </style>
    """, unsafe_allow_html=True)

# ============================================================
# 🔍 ADVANCED KEYWORD EXTRACTION (FIXED)
# ============================================================

def build_dynamic_stopwords(brand_names: List[str]) -> Set[str]:
    """Build comprehensive stopwords including brand names"""
    stopwords = CHINESE_STOPWORDS.copy()
    stopwords.update(ENGLISH_STOPWORDS)
    
    # Add brand names and variations
    for brand in brand_names:
        if brand:
            brand_lower = brand.lower().strip()
            stopwords.add(brand_lower)
            stopwords.add(brand.upper())
            stopwords.add(brand.title())
            # Add variations
            stopwords.add(f"{brand_lower}家")
            stopwords.add(f"{brand_lower}的")
            stopwords.add(f"{brand_lower}牌")
    
    return stopwords

def extract_keywords_fixed(text: str, stopwords: Set[str]) -> List[str]:
    """FIXED: Extract keywords accurately from text"""
    text = str(text).strip().lower()
    
    if not text:
        return []
    
    keywords = []
    
    if JIEBA_AVAILABLE:
        # Use jieba for accurate Chinese segmentation
        words = jieba.cut(text, cut_all=False)
        
        for word in words:
            word = word.strip()
            
            # Filter conditions
            if len(word) < 2:  # Too short
                continue
            if word in stopwords:  # Stopword
                continue
            if re.match(r'^\d+$', word):  # Pure numbers
                continue
            if re.match(r'^[a-z]$', word):  # Single letter
                continue
            
            keywords.append(word)
    else:
        # Fallback: regex-based extraction
        chinese_words = re.findall(r'[\u4e00-\u9fff]{2,6}', text)
        english_words = re.findall(r'[a-z]{3,20}', text)
        
        all_words = chinese_words + english_words
        
        for word in all_words:
            word = word.strip()
            if word and word not in stopwords and len(word) >= 2:
                keywords.append(word)
    
    return keywords

def categorize_keyword(keyword: str) -> Tuple[str, str]:
    """Categorize keyword into business dimensions"""
    kw_lower = keyword.lower()
    
    for category, info in CATEGORY_FRAMEWORK.items():
        for term in info["keywords"]:
            if term.lower() in kw_lower or kw_lower in term.lower():
                return category, ""
    
    return "📌 其他", ""

def extract_top_keywords_enhanced(
    posts: List[str],
    brand_names: List[str],
    min_frequency: int = 2,
    top_n: int = 30
) -> List[Dict[str, Any]]:
    """ENHANCED: Extract top keywords with accurate frequency counting"""
    
    if not posts:
        return []
    
    stopwords = build_dynamic_stopwords(brand_names)
    
    # Extract keywords from all posts
    keyword_posts_map = defaultdict(set)
    
    for idx, post in enumerate(posts):
        keywords = extract_keywords_fixed(post, stopwords)
        # Use set to count each keyword only once per post
        for kw in set(keywords):
            keyword_posts_map[kw].add(idx)
    
    # Count post-level frequency
    keyword_counts = {kw: len(post_ids) for kw, post_ids in keyword_posts_map.items()}
    
    # Filter by minimum frequency
    keyword_counts = {k: v for k, v in keyword_counts.items() if v >= min_frequency}
    
    # Rank keywords by frequency
    ranked = sorted(keyword_counts.items(), key=lambda x: -x[1])[:top_n]
    
    results = []
    for keyword, count in ranked:
        category, _ = categorize_keyword(keyword)
        
        # Get posts containing this keyword
        keyword_posts = [posts[idx] for idx in keyword_posts_map[keyword]]
        sentiment = calculate_sentiment_distribution(keyword_posts)
        
        # Determine priority
        if count >= 10:
            priority = "High"
            status = "🔴 High Priority"
        elif count >= 5:
            priority = "Medium"
            status = "🟠 Medium Priority"
        else:
            priority = "Low"
            status = "🟢 Low Priority"
        
        results.append({
            "keyword": keyword,
            "mentions": count,
            "category": category,
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
    """Analyze sentiment of single text"""
    text_lower = text.lower()
    
    pos_count = sum(1 for term in POSITIVE_TERMS if term in text_lower)
    neg_count = sum(1 for term in NEGATIVE_TERMS if term in text_lower)
    
    # Check for negation
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
    """Calculate sentiment distribution metrics"""
    if not posts:
        return {
            "positive": 0, "negative": 0, "neutral": 0, "total": 0,
            "positive_pct": 0, "negative_pct": 0, "neutral_pct": 0,
            "net_sentiment": 0, "confidence": "Insufficient data"
        }
    
    sentiments = [analyze_sentiment(post) for post in posts]
    counter = Counter(sentiments)
    total = len(sentiments)
    
    pos = counter.get("Positive", 0)
    neg = counter.get("Negative", 0)
    neu = counter.get("Neutral", 0)
    
    net_sentiment = (pos - neg) / total if total > 0 else 0
    
    # Confidence based on sample size
    if total < 10:
        confidence = "Low (n<10)"
    elif total < 30:
        confidence = "Medium (n<30)"
    else:
        confidence = "High (n≥30)"
    
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

# ============================================================
# 📊 WORD CLOUD GENERATION (FIXED)
# ============================================================

def create_word_cloud_fixed(
    posts: List[str],
    brand_names: List[str],
    title: str = "关键词词云"
):
    """FIXED: Create accurate word cloud from posts"""
    if not WORDCLOUD_AVAILABLE or not posts:
        return None
    
    # Build stopwords
    stopwords = build_dynamic_stopwords(brand_names)
    
    # Extract all keywords
    all_keywords = []
    for post in posts:
        keywords = extract_keywords_fixed(post, stopwords)
        all_keywords.extend(keywords)
    
    if not all_keywords:
        return None
    
    # Count frequency
    keyword_freq = Counter(all_keywords)
    
    # Generate word cloud
    try:
        # Try to find Chinese font
        font_paths = [
            "/System/Library/Fonts/STHeiti Medium.ttc",  # macOS
            "/System/Library/Fonts/PingFang.ttc",  # macOS
            "C:\\Windows\\Fonts\\msyh.ttc",  # Windows
            "C:\\Windows\\Fonts\\simhei.ttf",  # Windows
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",  # Linux
            "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",  # Linux
        ]
        
        font_path = None
        for path in font_paths:
            if os.path.exists(path):
                font_path = path
                break
        
        wc = WordCloud(
            width=1200,
            height=600,
            background_color='white',
            font_path=font_path,
            max_words=100,
            relative_scaling=0.5,
            colormap='Blues',  # Professional color scheme
            prefer_horizontal=0.7,
            min_font_size=12,
            max_font_size=120,
            collocations=False  # Prevent duplicate phrases
        ).generate_from_frequencies(keyword_freq)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 7), facecolor='white')
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(title, fontsize=18, pad=20, weight='bold', color='#1a1a1a')
        
        plt.tight_layout(pad=1)
        
        return fig
        
    except Exception as e:
        st.warning(f"词云生成失败: {str(e)}")
        return None

# ============================================================
# 📈 PROFESSIONAL VISUALIZATIONS
# ============================================================

def create_sentiment_gauge(sentiment_data: Dict, brand_name: str):
    """Professional sentiment gauge chart"""
    if not PLOTLY_AVAILABLE:
        return None
    
    net_sentiment = sentiment_data.get("net_sentiment", 0)
    gauge_value = (net_sentiment + 1) * 50  # Map -1~1 to 0~100
    
    # Color based on sentiment
    if net_sentiment > 0.3:
        color = "#27ae60"
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
            'text': f"<b>{brand_name}</b><br><span style='font-size:0.9em;color:#6c757d'>Net Sentiment Index</span>",
            'font': {'size': 18, 'family': 'Source Sans Pro'}
        },
        delta = {'reference': 50, 'increasing': {'color': "#27ae60"}},
        number = {'font': {'size': 40, 'family': 'Source Sans Pro'}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#e9ecef"},
            'bar': {'color': color, 'thickness': 0.25},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#e9ecef",
            'steps': [
                {'range': [0, 25], 'color': '#ffebee'},
                {'range': [25, 40], 'color': '#fff3e0'},
                {'range': [40, 60], 'color': '#f1f8e9'},
                {'range': [60, 75], 'color': '#e8f5e9'},
                {'range': [75, 100], 'color': '#c8e6c9'}
            ],
            'threshold': {
                'line': {'color': "#6c757d", 'width': 3},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(
        height=350,
        font={'family': "Source Sans Pro"},
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=30, r=30, t=80, b=30)
    )
    
    return fig

def create_keyword_frequency_chart(keyword_data: List[Dict], top_n: int = 15):
    """Professional horizontal bar chart for keyword frequency"""
    if not PLOTLY_AVAILABLE or not keyword_data:
        return None
    
    df = pd.DataFrame(keyword_data).head(top_n)
    df = df.sort_values('mentions', ascending=True)
    
    # Color by priority
    colors = []
    for priority in df['priority']:
        if priority == 'High':
            colors.append('#e74c3c')
        elif priority == 'Medium':
            colors.append('#f39c12')
        else:
            colors.append('#3498db')
    
    fig = go.Figure(go.Bar(
        x=df['mentions'],
        y=df['keyword'],
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(color='white', width=1)
        ),
        text=df['mentions'],
        textposition='outside',
        textfont=dict(size=11, family='Source Sans Pro', color='#1a1a1a'),
        hovertemplate='<b>%{y}</b><br>Mentions: %{x}<br>Category: %{customdata}<extra></extra>',
        customdata=df['category']
    ))
    
    fig.update_layout(
        title=dict(
            text="<b>Top Keywords by Mention Frequency</b>",
            font=dict(size=16, family='Source Sans Pro', color='#1a1a1a'),
            x=0,
            xanchor='left'
        ),
        xaxis=dict(
            title="Number of Mentions",
            showgrid=True,
            gridcolor='#f1f3f5',
            zeroline=False
        ),
        yaxis=dict(
            title="",
            showgrid=False
        ),
        height=max(400, len(df) * 30),
        margin=dict(l=150, r=50, t=60, b=50),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='white',
        font=dict(family='Source Sans Pro', size=12)
    )
    
    return fig

def create_sentiment_breakdown_chart(sentiment_data: Dict):
    """Professional donut chart for sentiment breakdown"""
    if not PLOTLY_AVAILABLE:
        return None
    
    labels = ['Positive', 'Neutral', 'Negative']
    values = [
        sentiment_data['positive'],
        sentiment_data['neutral'],
        sentiment_data['negative']
    ]
    colors = ['#27ae60', '#95a5a6', '#e74c3c']
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.5,
        marker=dict(colors=colors, line=dict(color='white', width=2)),
        textposition='outside',
        textinfo='label+percent',
        textfont=dict(size=13, family='Source Sans Pro'),
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        title=dict(
            text="<b>Sentiment Distribution</b>",
            font=dict(size=16, family='Source Sans Pro', color='#1a1a1a'),
            x=0.5,
            xanchor='center'
        ),
        height=400,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5,
            font=dict(size=12, family='Source Sans Pro')
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Source Sans Pro'),
        margin=dict(l=50, r=50, t=60, b=50)
    )
    
    return fig

def create_category_distribution(keyword_data: List[Dict]):
    """Professional sunburst chart for category distribution"""
    if not PLOTLY_AVAILABLE or not keyword_data:
        return None
    
    # Aggregate by category
    category_counts = defaultdict(int)
    category_keywords = defaultdict(list)
    
    for item in keyword_data:
        cat = item['category']
        category_counts[cat] += item['mentions']
        category_keywords[cat].append({
            'keyword': item['keyword'],
            'mentions': item['mentions']
        })
    
    # Build hierarchical data
    labels = ["Brand Discussion"]
    parents = [""]
    values = [0]
    colors = []
    
    color_map = {
        "🎨 产品体验": "#3498db",
        "💰 性价比": "#2ecc71",
        "📦 包装设计": "#9b59b6",
        "🚚 物流配送": "#f39c12",
        "🛡️ 售后服务": "#e74c3c",
        "⚠️ 真伪问题": "#c0392b",
        "🔄 竞品对比": "#34495e",
        "📌 其他": "#95a5a6"
    }
    
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        labels.append(cat)
        parents.append("Brand Discussion")
        values.append(count)
        colors.append(color_map.get(cat, "#95a5a6"))
        
        # Add top 3 keywords per category
        for kw in sorted(category_keywords[cat], key=lambda x: -x['mentions'])[:3]:
            labels.append(kw['keyword'])
            parents.append(cat)
            values.append(kw['mentions'])
            colors.append(color_map.get(cat, "#95a5a6"))
    
    fig = go.Figure(go.Sunburst(
        labels=labels,
        parents=parents,
        values=values,
        marker=dict(colors=colors, line=dict(color='white', width=2)),
        branchvalues="total",
        hovertemplate='<b>%{label}</b><br>Mentions: %{value}<br>Percentage: %{percentParent}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text="<b>Brand Discussion Breakdown</b>",
            font=dict(size=16, family='Source Sans Pro', color='#1a1a1a'),
            x=0.5,
            xanchor='center'
        ),
        height=550,
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Source Sans Pro', size=12),
        margin=dict(l=50, r=50, t=60, b=50)
    )
    
    return fig

def create_priority_matrix(keyword_data: List[Dict]):
    """Professional scatter plot for priority matrix"""
    if not PLOTLY_AVAILABLE or not keyword_data:
        return None
    
    df = pd.DataFrame(keyword_data)
    
    # Color by priority
    color_map = {
        'High': '#e74c3c',
        'Medium': '#f39c12',
        'Low': '#3498db'
    }
    
    fig = px.scatter(
        df,
        x='mentions',
        y='sentiment_score',
        size='mentions',
        color='priority',
        hover_name='keyword',
        hover_data={
            'category': True,
            'mentions': True,
            'sentiment_score': ':.2f',
            'priority': False
        },
        color_discrete_map=color_map,
        size_max=60
    )
    
    # Add quadrant lines
    max_mentions = df['mentions'].max()
    fig.add_hline(y=0, line_dash="dash", line_color="#6c757d", opacity=0.3)
    fig.add_vline(x=5, line_dash="dash", line_color="#6c757d", opacity=0.3)
    
    # Quadrant labels
    fig.add_annotation(
        x=max_mentions * 0.8, y=0.7,
        text="High Freq + Positive<br>Amplify Strengths",
        showarrow=False,
        font=dict(size=10, color="#27ae60"),
        bgcolor="rgba(255,255,255,0.8)",
        borderpad=4
    )
    
    fig.add_annotation(
        x=max_mentions * 0.8, y=-0.7,
        text="High Freq + Negative<br>Critical Focus",
        showarrow=False,
        font=dict(size=10, color="#e74c3c"),
        bgcolor="rgba(255,255,255,0.8)",
        borderpad=4
    )
    
    fig.update_layout(
        title=dict(
            text="<b>Strategic Priority Matrix</b>",
            font=dict(size=16, family='Source Sans Pro', color='#1a1a1a'),
            x=0,
            xanchor='left'
        ),
        xaxis=dict(
            title="Mention Frequency",
            showgrid=True,
            gridcolor='#f1f3f5'
        ),
        yaxis=dict(
            title="Sentiment Score (-1 to +1)",
            showgrid=True,
            gridcolor='#f1f3f5',
            zeroline=True,
            zerolinecolor='#6c757d'
        ),
        height=500,
        showlegend=True,
        legend=dict(
            title="Priority Level",
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="right",
            x=1.15,
            font=dict(size=11, family='Source Sans Pro')
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='white',
        font=dict(family='Source Sans Pro', size=12),
        margin=dict(l=60, r=150, t=60, b=60)
    )
    
    return fig

# ============================================================
# 📱 STREAMLIT APP
# ============================================================

st.set_page_config(
    page_title="Brand Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply professional UI
apply_cision_ui()

# Header
st.markdown('<div class="main-header">📊 Brand Intelligence Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Professional Consulting-Grade Analytics · Strategic Insights · Data-Driven Decisions</div>', unsafe_allow_html=True)

# ============================================================
# SIDEBAR CONFIGURATION
# ============================================================

with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    data_source = st.selectbox(
        "Data Source",
        ["🔍 Xiaohongshu", "🛒 E-commerce Reviews", "📱 Social Media", "💬 Customer Feedback"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### 🏢 Brand Settings")
    
    primary_brand = st.text_input("Primary Brand", value="YSL", help="Enter your brand name")
    
    enable_competitor = st.checkbox("Enable Competitor Analysis", value=False)
    
    if enable_competitor:
        competitor_brand = st.text_input("Competitor Brand", value="Dior")
    else:
        competitor_brand = ""
    
    st.markdown("---")
    
    with st.expander("🔧 Advanced Settings"):
        min_frequency = st.slider("Min Keyword Frequency", 1, 5, 2, help="Minimum times a keyword must appear")
        top_n_keywords = st.slider("Top N Keywords", 10, 50, 30, help="Number of keywords to display")
        enable_wordcloud = st.checkbox("Enable Word Cloud", value=True)
    
    st.markdown("---")
    st.markdown("### 📊 System Status")
    
    if PLOTLY_AVAILABLE:
        st.success("✅ Plotly Charts")
    else:
        st.error("❌ Plotly Missing")
        
    if JIEBA_AVAILABLE:
        st.success("✅ Chinese Segmentation")
    else:
        st.warning("⚠️ Jieba Missing")
        
    if WORDCLOUD_AVAILABLE:
        st.success("✅ Word Cloud")
    else:
        st.warning("⚠️ WordCloud Missing")

# ============================================================
# SESSION STATE
# ============================================================

if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

# ============================================================
# MAIN TABS
# ============================================================

tab_input, tab_overview, tab_keywords, tab_sentiment, tab_insights, tab_export = st.tabs([
    "📥 Data Input",
    "📊 Overview",
    "🔑 Keyword Analysis",
    "😊 Sentiment Analysis",
    "💡 Strategic Insights",
    "📄 Export Report"
])

# ============================================================
# TAB 1: DATA INPUT
# ============================================================

with tab_input:
    st.markdown('<div class="section-header">Data Input</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <div class="info-box-title">📖 Instructions</div>
        Upload your brand comments data via CSV or paste manually. The system will automatically clean, deduplicate, and analyze the data.<br>
        <strong>Recommended sample size:</strong> 30+ comments for reliable insights.
    </div>
    """, unsafe_allow_html=True)
    
    def load_brand_data(label: str, key: str) -> List[str]:
        """Load data for a brand"""
        st.markdown(f"#### {label}")
        
        input_method = st.radio(
            "Input Method",
            ["📁 CSV Upload", "✍️ Manual Input"],
            horizontal=True,
            key=f"{key}_method"
        )
        
        posts = []
        
        if input_method == "📁 CSV Upload":
            uploaded_file = st.file_uploader(
                f"Upload CSV for {label}",
                type=["csv"],
                key=f"{key}_file",
                help="CSV must contain a 'text' column with comments"
            )
            
            if uploaded_file:
                try:
                    df = pd.read_csv(uploaded_file)
                    
                    # Let user select text column
                    text_col = st.selectbox(
                        "Select text column",
                        df.columns.tolist(),
                        key=f"{key}_col"
                    )
                    
                    raw_posts = df[text_col].dropna().astype(str).tolist()
                    
                    # Clean and deduplicate
                    seen = set()
                    for post in raw_posts:
                        post = post.strip()
                        if len(post) >= 5 and post not in seen:
                            posts.append(post)
                            seen.add(post)
                    
                    # Show metrics
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Raw Rows", len(raw_posts))
                    col2.metric("Clean Posts", len(posts))
                    col3.metric("Removed", len(raw_posts) - len(posts))
                    
                except Exception as e:
                    st.error(f"❌ Failed to load: {str(e)}")
        
        else:
            # Manual input
            raw_text = st.text_area(
                "Paste comments (one per line)",
                height=250,
                key=f"{key}_textarea",
                placeholder="Enter comments here...\nOne comment per line"
            )
            
            if raw_text.strip():
                lines = raw_text.strip().split('\n')
                seen = set()
                for line in lines:
                    line = line.strip()
                    if len(line) >= 5 and line not in seen:
                        posts.append(line)
                        seen.add(line)
                
                if posts:
                    st.success(f"✅ Loaded {len(posts)} comments")
        
        return posts
    
    # Load data for primary and competitor
    col_brand1, col_brand2 = st.columns(2)
    
    with col_brand1:
        primary_posts = load_brand_data(f"🎯 {primary_brand}", "primary")
    
    with col_brand2:
        if enable_competitor:
            competitor_posts = load_brand_data(f"🔄 {competitor_brand}", "competitor")
        else:
            competitor_posts = []

# ============================================================
# TAB 2: OVERVIEW
# ============================================================

with tab_overview:
    st.markdown('<div class="section-header">Analysis Overview</div>', unsafe_allow_html=True)
    
    if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
        if 'primary_posts' not in locals() or not primary_posts:
            st.error("⚠️ Please provide primary brand data in the Data Input tab")
            st.stop()
        
        with st.spinner("🔄 Analyzing data..."):
            brand_names = [primary_brand]
            if enable_competitor and competitor_brand:
                brand_names.append(competitor_brand)
            
            # Analyze primary brand
            primary_keywords = extract_top_keywords_enhanced(
                primary_posts,
                brand_names,
                min_frequency=min_frequency,
                top_n=top_n_keywords
            )
            primary_sentiment = calculate_sentiment_distribution(primary_posts)
            
            # Analyze competitor if enabled
            if enable_competitor and competitor_posts:
                competitor_keywords = extract_top_keywords_enhanced(
                    competitor_posts,
                    brand_names,
                    min_frequency=min_frequency,
                    top_n=top_n_keywords
                )
                competitor_sentiment = calculate_sentiment_distribution(competitor_posts)
            else:
                competitor_keywords = []
                competitor_sentiment = {}
            
            # Store results
            st.session_state.analysis_results = {
                "primary": {
                    "brand": primary_brand,
                    "posts": primary_posts,
                    "keywords": primary_keywords,
                    "sentiment": primary_sentiment
                },
                "competitor": {
                    "brand": competitor_brand,
                    "posts": competitor_posts,
                    "keywords": competitor_keywords,
                    "sentiment": competitor_sentiment
                } if enable_competitor and competitor_posts else None,
                "brand_names": brand_names
            }
            
            st.success("✅ Analysis completed!")
            st.rerun()
    
    # Display results
    if st.session_state.analysis_results:
        results = st.session_state.analysis_results
        primary_data = results["primary"]
        
        st.markdown("---")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Sample Size</div>
                <div class="metric-value">{len(primary_data['posts'])}</div>
                <div class="metric-delta">{primary_data['sentiment']['confidence']}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            net_sent = primary_data['sentiment']['net_sentiment']
            color = "#27ae60" if net_sent > 0.2 else "#e74c3c" if net_sent < -0.2 else "#f39c12"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Net Sentiment</div>
                <div class="metric-value" style="color:{color};">{net_sent:.2f}</div>
                <div class="metric-delta">{primary_data['sentiment']['positive_pct']:.0%} Positive</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            high_priority = sum(1 for kw in primary_data['keywords'] if kw['priority'] == 'High')
            color = "#e74c3c" if high_priority > 0 else "#27ae60"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">High Priority</div>
                <div class="metric-value" style="color:{color};">{high_priority}</div>
                <div class="metric-delta">Requires attention</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Keywords Identified</div>
                <div class="metric-value">{len(primary_data['keywords'])}</div>
                <div class="metric-delta">Top {top_n_keywords}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Sentiment Gauge
        if PLOTLY_AVAILABLE:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            fig_gauge = create_sentiment_gauge(primary_data['sentiment'], primary_brand)
            if fig_gauge:
                st.plotly_chart(fig_gauge, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# TAB 3: KEYWORD ANALYSIS
# ============================================================

with tab_keywords:
    st.markdown('<div class="section-header">Keyword Analysis</div>', unsafe_allow_html=True)
    
    if not st.session_state.analysis_results:
        st.info("👈 Please run analysis in the Overview tab first")
        st.stop()
    
    results = st.session_state.analysis_results
    primary_data = results["primary"]
    
    st.markdown("""
    <div class="info-box">
        <div class="info-box-title">📖 How to Read</div>
        <strong>Mention Frequency:</strong> Number of comments containing this keyword<br>
        <strong>Category:</strong> Business dimension (Product Experience, Value, Packaging, etc.)<br>
        <strong>Sentiment Score:</strong> -1 (completely negative) to +1 (completely positive)<br>
        <strong>Priority:</strong> Based on frequency - High (≥10), Medium (5-9), Low (2-4)
    </div>
    """, unsafe_allow_html=True)
    
    # Word Cloud
    if enable_wordcloud and WORDCLOUD_AVAILABLE:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Keyword Word Cloud</div>', unsafe_allow_html=True)
        
        wc_fig = create_word_cloud_fixed(
            primary_data['posts'],
            results['brand_names'],
            f"{primary_brand} - Key Discussion Topics"
        )
        
        if wc_fig:
            st.pyplot(wc_fig)
            plt.close()
        else:
            st.info("No keywords found for word cloud generation")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Keyword Frequency Chart
    if PLOTLY_AVAILABLE and primary_data['keywords']:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        fig_freq = create_keyword_frequency_chart(primary_data['keywords'], top_n=15)
        if fig_freq:
            st.plotly_chart(fig_freq, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Priority Matrix
    if PLOTLY_AVAILABLE and primary_data['keywords']:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        fig_matrix = create_priority_matrix(primary_data['keywords'])
        if fig_matrix:
            st.plotly_chart(fig_matrix, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Brand Discussion Breakdown (Sunburst)
    if PLOTLY_AVAILABLE and primary_data['keywords']:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        fig_sunburst = create_category_distribution(primary_data['keywords'])
        if fig_sunburst:
            st.plotly_chart(fig_sunburst, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Keyword Table
    st.markdown("---")
    st.markdown("### Detailed Keyword Analysis")
    
    if primary_data['keywords']:
        df_keywords = pd.DataFrame(primary_data['keywords'])
        
        st.dataframe(
            df_keywords[[
                "keyword", "mentions", "category", "priority",
                "sentiment_score", "positive_ratio", "negative_ratio"
            ]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "keyword": st.column_config.TextColumn("Keyword", width="medium"),
                "mentions": st.column_config.NumberColumn("Mentions", format="%d"),
                "category": st.column_config.TextColumn("Category", width="medium"),
                "priority": st.column_config.TextColumn("Priority", width="small"),
                "sentiment_score": st.column_config.ProgressColumn(
                    "Sentiment",
                    format="%.2f",
                    min_value=-1,
                    max_value=1
                ),
                "positive_ratio": st.column_config.ProgressColumn(
                    "Positive %",
                    format="%.0%",
                    min_value=0,
                    max_value=1
                ),
                "negative_ratio": st.column_config.ProgressColumn(
                    "Negative %",
                    format="%.0%",
                    min_value=0,
                    max_value=1
                )
            }
        )

# ============================================================
# TAB 4: SENTIMENT ANALYSIS
# ============================================================

with tab_sentiment:
    st.markdown('<div class="section-header">Sentiment Analysis</div>', unsafe_allow_html=True)
    
    if not st.session_state.analysis_results:
        st.info("👈 Please run analysis in the Overview tab first")
        st.stop()
    
    results = st.session_state.analysis_results
    primary_data = results["primary"]
    
    st.markdown("""
    <div class="info-box">
        <div class="info-box-title">📖 Methodology</div>
        Sentiment analysis uses rule-based classification with 200+ positive and negative term dictionaries.<br>
        <strong>Net Sentiment = (Positive - Negative) / Total</strong>, ranging from -1 to +1<br>
        <strong>Confidence level</strong> depends on sample size: &lt;10 (Low), 10-30 (Medium), ≥30 (High)
    </div>
    """, unsafe_allow_html=True)
    
    # Sentiment breakdown
    col_sent1, col_sent2 = st.columns(2)
    
    with col_sent1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        if PLOTLY_AVAILABLE:
            fig_donut = create_sentiment_breakdown_chart(primary_data['sentiment'])
            if fig_donut:
                st.plotly_chart(fig_donut, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_sent2:
        sent_data = primary_data['sentiment']
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Positive Comments</div>
            <div class="metric-value" style="color:#27ae60;">{sent_data['positive_pct']:.1%}</div>
            <div class="metric-delta">{sent_data['positive']} comments</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Neutral Comments</div>
            <div class="metric-value" style="color:#95a5a6;">{sent_data['neutral_pct']:.1%}</div>
            <div class="metric-delta">{sent_data['neutral']} comments</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Negative Comments</div>
            <div class="metric-value" style="color:#e74c3c;">{sent_data['negative_pct']:.1%}</div>
            <div class="metric-delta">{sent_data['negative']} comments</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Sentiment interpretation
    st.markdown("---")
    st.markdown("### Strategic Interpretation")
    
    net_sent = sent_data['net_sentiment']
    
    if net_sent > 0.3:
        st.markdown("""
        <div class="info-box" style="border-left-color:#27ae60;">
            <div class="info-box-title" style="color:#27ae60;">✅ Strong Positive Signal</div>
            Net Sentiment: {:.1%} | Confidence: {}<br>
            <strong>Recommendation:</strong> Amplify positive reviews in marketing campaigns. Feature customer testimonials.
        </div>
        """.format(net_sent, sent_data['confidence']), unsafe_allow_html=True)
    elif net_sent > 0:
        st.markdown("""
        <div class="info-box" style="border-left-color:#f39c12;">
            <div class="info-box-title" style="color:#f39c12;">ℹ️ Moderately Positive</div>
            Net Sentiment: {:.1%} | Confidence: {}<br>
            <strong>Recommendation:</strong> Maintain current strategy. Address negative drivers to improve further.
        </div>
        """.format(net_sent, sent_data['confidence']), unsafe_allow_html=True)
    elif net_sent > -0.2:
        st.markdown("""
        <div class="info-box" style="border-left-color:#e67e22;">
            <div class="info-box-title" style="color:#e67e22;">⚠️ Mixed Sentiment</div>
            Net Sentiment: {:.1%} | Confidence: {}<br>
            <strong>Recommendation:</strong> Investigate negative causes. Enhance positive touchpoints.
        </div>
        """.format(net_sent, sent_data['confidence']), unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="info-box" style="border-left-color:#e74c3c;">
            <div class="info-box-title" style="color:#e74c3c;">🚨 Requires Attention</div>
            Net Sentiment: {:.1%} | Confidence: {}<br>
            <strong>Recommendation:</strong> Immediate action required. Address critical negative issues.
        </div>
        """.format(net_sent, sent_data['confidence']), unsafe_allow_html=True)

# ============================================================
# TAB 5: STRATEGIC INSIGHTS
# ============================================================

with tab_insights:
    st.markdown('<div class="section-header">Strategic Insights</div>', unsafe_allow_html=True)
    
    if not st.session_state.analysis_results:
        st.info("👈 Please run analysis in the Overview tab first")
        st.stop()
    
    results = st.session_state.analysis_results
    primary_data = results["primary"]
    
    st.markdown("""
    <div class="info-box">
        <div class="info-box-title">📖 Priority Framework</div>
        System automatically generates prioritized recommendations based on keyword frequency and sentiment.<br>
        <strong>High Priority (≥10 mentions):</strong> Action within 24-48h<br>
        <strong>Medium Priority (5-9 mentions):</strong> Plan response within 1 week<br>
        <strong>Low Priority (2-4 mentions):</strong> Monitor and track over 2 weeks
    </div>
    """, unsafe_allow_html=True)
    
    # Categorize by priority
    high_priority = [kw for kw in primary_data['keywords'] if kw['priority'] == 'High']
    medium_priority = [kw for kw in primary_data['keywords'] if kw['priority'] == 'Medium']
    low_priority = [kw for kw in primary_data['keywords'] if kw['priority'] == 'Low']
    
    # High priority items
    if high_priority:
        st.markdown("### 🔴 High Priority Items")
        
        for kw in high_priority:
            sentiment_color = "#27ae60" if kw['sentiment_score'] > 0.2 else "#e74c3c" if kw['sentiment_score'] < -0.2 else "#f39c12"
            
            st.markdown(f"""
            <div class="metric-card" style="border-left:4px solid {sentiment_color};">
                <h4 style="margin:0 0 0.5rem 0;color:{sentiment_color};">{kw['keyword']} ({kw['category']})</h4>
                <strong>Mentions:</strong> {kw['mentions']} times<br>
                <strong>Sentiment:</strong> {kw['sentiment_score']:.2f} (Positive: {kw['positive_ratio']:.0%}, Negative: {kw['negative_ratio']:.0%})<br>
                <strong>Status:</strong> {kw['status']}<br>
                <strong>Action:</strong> Immediate response required within 24-48 hours
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("✅ No high priority issues identified")
    
    # Medium priority
    if medium_priority:
        st.markdown("---")
        st.markdown("### 🟠 Medium Priority Items")
        
        cols = st.columns(2)
        for idx, kw in enumerate(medium_priority):
            with cols[idx % 2]:
                sentiment_icon = "😊" if kw['sentiment_score'] > 0.2 else "😞" if kw['sentiment_score'] < -0.2 else "😐"
                
                st.markdown(f"""
                <div class="metric-card" style="border-left:4px solid #f39c12;">
                    <h5 style="margin:0 0 0.5rem 0;">{sentiment_icon} {kw['keyword']}</h5>
                    <strong>Mentions:</strong> {kw['mentions']}<br>
                    <strong>Sentiment:</strong> {kw['sentiment_score']:.2f}<br>
                    <strong>Action:</strong> Plan response within 1 week
                </div>
                """, unsafe_allow_html=True)
    
    # Low priority (collapsible)
    if low_priority:
        with st.expander(f"🟢 Low Priority Items ({len(low_priority)}) - Click to expand"):
            for kw in low_priority:
                st.markdown(f"- **{kw['keyword']}** ({kw['category']}): {kw['mentions']} mentions - Monitor and track")

# ============================================================
# TAB 6: EXPORT REPORT
# ============================================================

with tab_export:
    st.markdown('<div class="section-header">Export Report</div>', unsafe_allow_html=True)
    
    if not st.session_state.analysis_results:
        st.info("👈 Please run analysis in the Overview tab first")
        st.stop()
    
    results = st.session_state.analysis_results
    primary_data = results["primary"]
    
    st.markdown("""
    <div class="info-box">
        <div class="info-box-title">📖 Export Options</div>
        Export analysis results in CSV or JSON format for further processing or reporting.<br>
        <strong>CSV:</strong> Suitable for Excel viewing and sharing with marketing teams<br>
        <strong>JSON:</strong> Suitable for technical teams and system integration
    </div>
    """, unsafe_allow_html=True)
    
    # Prepare export data
    keywords_df = pd.DataFrame(primary_data['keywords'])
    
    summary_data = {
        "Brand": [primary_brand],
        "Sample Size": [len(primary_data['posts'])],
        "Confidence": [primary_data['sentiment']['confidence']],
        "Net Sentiment": [primary_data['sentiment']['net_sentiment']],
        "Positive %": [primary_data['sentiment']['positive_pct']],
        "Negative %": [primary_data['sentiment']['negative_pct']],
        "High Priority Items": [sum(1 for kw in primary_data['keywords'] if kw['priority'] == 'High')]
    }
    summary_df = pd.DataFrame(summary_data)
    
    # Download buttons
    col_dl1, col_dl2, col_dl3 = st.columns(3)
    
    with col_dl1:
        st.download_button(
            label="📥 Download Keywords CSV",
            data=keywords_df.to_csv(index=False).encode('utf-8-sig'),
            file_name=f"{primary_brand}_keywords_{date.today()}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col_dl2:
        st.download_button(
            label="📥 Download Summary CSV",
            data=summary_df.to_csv(index=False).encode('utf-8-sig'),
            file_name=f"{primary_brand}_summary_{date.today()}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col_dl3:
        json_export = json.dumps(results, ensure_ascii=False, indent=2, default=str)
        st.download_button(
            label="📥 Download Full JSON",
            data=json_export.encode('utf-8'),
            file_name=f"{primary_brand}_analysis_{date.today()}.json",
            mime="application/json",
            use_container_width=True
        )
    
    # Preview
    st.markdown("---")
    
    with st.expander("📊 Preview Keywords Analysis"):
        st.dataframe(keywords_df, use_container_width=True, hide_index=True)
    
    with st.expander("📋 Preview Executive Summary"):
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

col_footer1, col_footer2, col_footer3 = st.columns(3)

with col_footer1:
    st.markdown("**🎯 Core Features**")
    st.markdown("""
    - Advanced Keyword Extraction
    - Sentiment Analysis
    - Priority Framework
    - Visual Analytics
    - Export Capabilities
    """)

with col_footer2:
    st.markdown("**📊 Analysis Dimensions**")
    st.markdown("""
    - Product Experience
    - Value Perception
    - Package Design
    - Logistics & Delivery
    - Customer Service
    - Authenticity
    - Competitive Position
    """)

with col_footer3:
    st.markdown("**💡 Best Practices**")
    st.markdown("""
    - Sample size: 30+ comments
    - Update frequency: Weekly
    - Focus on high priority
    - Track sentiment trends
    - Monitor key categories
    """)

st.markdown("---")
st.caption("**Professional Brand Intelligence Platform** | Consulting-Grade Analytics · Strategic Insights · Data-Driven Decisions")
