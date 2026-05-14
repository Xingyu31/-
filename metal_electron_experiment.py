"""
金属电子逸出功与质荷比测定实验数据处理系统
基于Streamlit开发的交互式数据处理网页

===== 关键优化：st.form 模式 =====
1. 所有数据输入使用 st.form 包裹，消除输入时的跳闪问题
2. 用户输入时不触发重运行，点击"保存数据"按钮才一次性提交所有修改
3. 动态增删行按钮放在form外（操作不频繁，可接受跳闪）

===== 物理单位正确性 =====
严格执行 μA → A 单位转换，确保计算结果准确
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
from io import BytesIO
import base64

# ==================== 物理常数与单位说明 ====================
# 理查森公式：lg(I/T²) = lg(AS) - 5040 φ / T
# 其中：
#   I - 发射电流，单位安培(A)
#   T - 灯丝温度，单位开尔文(K)
#   A - 发射常数
#   S - 阴极面积，单位cm²
#   φ - 逸出电势，单位V
#   W₀ = eφ - 逸出功，单位eV
#
# ===== 重要：单位转换 =====
# 用户输入的 Ia 单位是 μA（微安）
# 计算公式需要 A（安培）
# 转换关系：1 μA = 10^-6 A
# ================================

# 电子荷质比理论值
THEORETICAL_E_M = 1.76e11  # C/kg
# 金属钨的逸出功理论值（更正：钨的实际理论值约4.54eV，原5.54eV有误）
THEORETICAL_W0 = 4.54  # eV

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="金属电子逸出功与质核比测定实验",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== 自定义样式 ====================
st.markdown("""
<style>
    .main {
        background-color: #f0f8ff;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #e6f2ff;
        border-radius: 5px 5px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        padding-left: 15px;
        padding-right: 15px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1a73e8;
        color: white;
    }
    .css-18e3th9 {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    .stButton>button {
        background-color: #1a73e8;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #1557b0;
        color: white;
    }
    .info-box {
        background-color: #e8f0fe;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #1a73e8;
    }
    .result-box {
        background-color: #f0fff4;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #28a745;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #ffc107;
    }
</style>
""", unsafe_allow_html=True)

# ==================== 内置数据 ====================
# 表3-6-1：灯丝电流与温度的对应关系
FILAMENT_CURRENTS = [0.600, 0.625, 0.650, 0.675, 0.700, 0.725, 0.750]
TEMPERATURES = [1.88, 1.92, 1.96, 2.00, 2.04, 2.08, 2.12]  # ×10³ K
ACCELERATING_VOLTAGES = [16.0, 25.0, 36.0, 49.0, 64.0, 81.0, 100.0]

# 荷质比测定参数
K_PRIME = 1.445e-2  # 1.445×10⁻²
A_COIL = 4.0  # mm, 线圈半径

# 示例数据（用于演示）- 单位：μA
SAMPLE_DATA_EMISSION = np.array([
    [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08],
    [0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.17],
    [0.12, 0.16, 0.20, 0.24, 0.28, 0.32, 0.36],
    [0.28, 0.36, 0.44, 0.52, 0.60, 0.68, 0.76],
    [0.60, 0.78, 0.96, 1.14, 1.32, 1.50, 1.68],
    [1.30, 1.68, 2.06, 2.44, 2.82, 3.20, 3.58],
    [2.80, 3.62, 4.44, 5.26, 6.08, 6.90, 7.72]
])

# ==================== Session State 初始化 ====================
def init_session_state():
    """初始化所有 session_state 变量 - 使用 DataFrame 而非 numpy 数组"""
    # 逸出功数据 - 使用 DataFrame 存储
    if 'emission_data' not in st.session_state:
        st.session_state.emission_data = pd.DataFrame(
            np.zeros((7, 7)),
            index=FILAMENT_CURRENTS,
            columns=ACCELERATING_VOLTAGES
        )

    # 示例数据加载标记
    if 'sample_loaded' not in st.session_state:
        st.session_state.sample_loaded = False

    # 荷质比数据
    if '荷质比_data' not in st.session_state:
        # 默认初始化空DataFrame
        st.session_state.荷质比_data = pd.DataFrame(
            columns=['励磁电流_Is'] + [f'Un={v}V' for v in [10, 15, 20, 25, 30]]
        )

    # 临界电流调整 - 初始化为空DataFrame而非空字典
    if '临界电流' not in st.session_state:
        st.session_state.临界电流 = pd.DataFrame(columns=['Un', 'Ic', 'Ic² (A²)'])

# 执行初始化
init_session_state()

# ==================== 辅助函数 ====================
def calculate_emission_work_function(df_emission):
    """
    计算电子逸出功

    ===== 单位说明 =====
    输入：Ia 单位为 μA
    计算：严格转换为 A 进行对数计算
    输出：逸出功单位为 eV
    ===================
    """
    results = {
        'lgIa_fits': [],
        'richardson_data': [],
        'work_function': None,
        'error': None
    }

    # 第一步：对每个温度绘制lgIa - √Ua图线
    sqrt_Ua = np.sqrt(ACCELERATING_VOLTAGES)

    I_values_A = []  # 零场发射电流，单位：安培(A)
    valid_temps = []

    for i, (If, T) in enumerate(zip(FILAMENT_CURRENTS, TEMPERATURES)):
        Ia_values_muA = df_emission.iloc[i].values
        if np.all(Ia_values_muA == 0):
            continue

        # 过滤掉零值
        mask = Ia_values_muA > 0
        if np.sum(mask) < 2:
            continue

        Ia_filtered_muA = Ia_values_muA[mask]
        sqrt_Ua_filtered = sqrt_Ua[mask]

        # ===== 关键修复：单位转换 μA → A =====
        Ia_filtered_A = Ia_filtered_muA * 1e-6  # 转换为安培
        lg_Ia = np.log10(Ia_filtered_A)  # 对安培取对数

        # 线性拟合
        slope, intercept, r_value, p_value, std_err = stats.linregress(sqrt_Ua_filtered, lg_Ia)

        # 零场时√Ua=0，截距就是 lg(I_A)，其中 I_A 单位为安培
        # I_A = 10^intercept (单位：A)
        I_A = 10 ** intercept  # 已经是安培单位

        # 转换回微安用于显示
        I_muA = I_A * 1e6

        results['lgIa_fits'].append({
            'T': T * 1000,
            'If': If,
            'sqrt_Ua': sqrt_Ua_filtered,
            'lg_Ia': lg_Ia,  # 基于安培的对数
            'slope': slope,
            'intercept': intercept,  # lg(I_A)
            'r_squared': r_value ** 2,
            'I_A': I_A,  # 零场电流（安培）
            'I_muA': I_muA,  # 零场电流（微安，用于显示）
            'Ia_data_muA': Ia_filtered_muA
        })

        I_values_A.append(I_A)  # 保存安培单位的零场电流
        valid_temps.append(T)

    if len(I_values_A) < 2:
        return None, "数据不足，无法进行理查森拟合（至少需要2组完整数据）"

    # 第二步：理查森直线法 lg(I/T²) - 1/T
    I_values_A = np.array(I_values_A)  # 单位：安培(A)
    T_kelvin = np.array(valid_temps) * 1000  # 转换为K

    # ===== 关键：I 已经是安培单位，直接计算 lg(I/T²) =====
    lg_I_over_T2 = np.log10(I_values_A / (T_kelvin ** 2))
    inv_T = 1 / T_kelvin

    # 线性拟合
    slope_rich, intercept_rich, r_rich, p_rich, std_err_rich = stats.linregress(inv_T, lg_I_over_T2)

    # 计算逸出电势 φ
    # lg(I/T²) = lg(AS) - 5040 * φ * (1/T)
    # slope = -5040 * φ
    phi = -slope_rich / 5040
    W0 = phi  # eV

    # 百分误差
    error_percent = abs(W0 - THEORETICAL_W0) / THEORETICAL_W0 * 100

    # 转换回微安用于显示
    I_values_muA = I_values_A * 1e6

    results['richardson_data'] = {
        'inv_T': inv_T,
        'lg_I_over_T2': lg_I_over_T2,
        'slope': slope_rich,
        'intercept': intercept_rich,
        'r_squared': r_rich ** 2,
        'phi': phi,
        'W0': W0,
        'error_percent': error_percent,
        'T_kelvin': T_kelvin,
        'I_values_A': I_values_A,  # 安培（内部计算用）
        'I_values_muA': I_values_muA  # 微安（显示用）
    }

    return results, None

def calculate_e_m_ratio(df_data):
    """计算电子荷质比"""
    results = {
        'curves': [],
        'critical_currents': {},
        'linear_fit': None,
        'e_m_ratio': None,
        'error_percent': None
    }

    # 提取数据
    Is_values = df_data['励磁电流_Is'].values.astype(float)
    Un_columns = [col for col in df_data.columns if 'Un=' in col]

    for col in Un_columns:
        Un = float(col.replace('Un=', '').replace('V', ''))
        Ia_values = df_data[col].values.astype(float)

        # 过滤有效数据
        mask = ~np.isnan(Ia_values) & (Ia_values >= 0)
        if np.sum(mask) < 5:
            continue

        Is_filtered = Is_values[mask]
        Ia_filtered = Ia_values[mask]

        results['curves'].append({
            'Un': Un,
            'Is': Is_filtered,
            'Ia': Ia_filtered
        })

    return results

def get_table_download_link(df, filename='data.csv', link_text='下载CSV文件'):
    """生成CSV下载链接"""
    csv = df.to_csv(index=False, encoding='utf-8-sig')
    b64 = base64.b64encode(csv.encode('utf-8-sig')).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" style="color: #1a73e8; text-decoration: none; font-weight: bold;">📥 {link_text}</a>'
    return href

# ==================== 主标题 ====================
st.title("⚡ 金属电子逸出功与质核比测定实验")
st.markdown("---")

# ==================== 导航标签页 ====================
tabs = st.tabs([
    "📚 实验说明",
    "📊 逸出功数据输入",
    "📈 逸出功数据处理",
    "⚡ 荷质比数据输入",
    "📉 荷质比数据处理",
    "📋 结果汇总与导出"
])

# ==================== 模块一：实验说明 ====================
with tabs[0]:
    st.header("实验说明")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        <div class="info-box">
        <h4>📋 实验目的</h4>
        <p>1. 了解热电子发射的基本规律，学习用理查森直线法测定金属电子逸出功</p>
        <p>2. 利用磁控法测定电子荷质比（e/m）</p>
        <p>3. 掌握数据处理与作图方法，提高实验数据分析能力</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 一、电子逸出功的测定（理查森直线法）")
        st.markdown("""
        根据理查森-杜什曼公式：
        $$I = AST^2 e^{-\\frac{e\\varphi}{kT}}$$
        
        取对数整理得：
        $$\\lg\\left(\\frac{I}{T^2}\\right) = \\lg(AS) - 5040\\varphi \\cdot \\frac{1}{T}$$
        
        通过测量不同温度下的零场发射电流，作 $\\lg(I/T^2) - 1/T$ 图线，由斜率求逸出电势 $\\varphi$。
        
        **注意：** I 的单位必须是 **安培(A)**，不能用微安(μA)直接计算！
        """)

        st.markdown("### 二、电子荷质比的测定（磁控法）")
        st.markdown("""
        利用磁场对电子束的偏转作用，当励磁电流达到临界值 $I_c$ 时，阳极电流急剧下降。
        
        电子荷质比计算公式：
        $$\\frac{e}{m} = \\frac{8K}{a^2 \\cdot K'^2}$$
        
        其中 $K$ 为 $U_n - I_c^2$ 图线的斜率。
        """)

    with col2:
        st.markdown("""
        <div class="info-box">
        <h4>📊 数据表格结构</h4>
        <p><strong>表3-6-1：</strong>灯丝电流-温度对应关系（内置）</p>
        <p><strong>表3-6-2：</strong>不同Ua和If下的Ia值（用户输入，单位μA）</p>
        <p><strong>表3-6-3：</strong>lgIa-√Ua拟合结果</p>
        <p><strong>表3-6-4：</strong>理查森直线法数据</p>
        <p><strong>表3-6-5：</strong>荷质比测定数据（用户输入）</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="warning-box">
        <h4>💡 使用提示</h4>
        <p>• 点击左侧按钮加载示例数据</p>
        <p>• <strong>输入数据后请点击"保存数据"按钮</strong></p>
        <p>• 图表支持缩放、平移</p>
        <p>• 结果可导出CSV格式</p>
        </div>
        """, unsafe_allow_html=True)

    # 显示内置的灯丝电流-温度对应表
    st.markdown("### 表3-6-1：灯丝电流与温度的对应关系（内置）")
    df_temp = pd.DataFrame({
        '灯丝电流If (A)': FILAMENT_CURRENTS + [0.775, 0.800],
        '温度T (×10³K)': TEMPERATURES + [2.16, 2.20]
    })
    st.dataframe(df_temp.style.highlight_max(axis=0), width='stretch')

    # 单位说明
    st.markdown("---")
    st.markdown("### ⚠️ 重要：单位说明")
    st.markdown("""
    <div class="warning-box">
    <h4>逸出功测定 - 单位转换说明</h4>
    <p><strong>输入：</strong>表格中输入的 Ia 单位为 <strong>微安(μA)</strong></p>
    <p><strong>计算：</strong>程序自动转换为 <strong>安培(A)</strong> 后进行对数计算</p>
    <p><strong>原因：</strong>理查森公式要求 I 的单位为安培，否则截距会偏移 lg(10⁻⁶) = -6，
    导致最终逸出功计算结果偏差</p>
    <p><strong>示例：</strong>I = 1 μA = 10⁻⁶ A，lg(I) = -6（正确），而非 0（错误）</p>
    </div>
    """, unsafe_allow_html=True)

# ==================== 模块二：电子逸出功测定数据输入（st.form模式） ====================
with tabs[1]:
    st.header("电子逸出功测定 - 数据输入")

    # ========== 使用 st.form 包裹，消除输入跳闪 ==========
    with st.form("逸出功数据表单"):
        col_info1, col_info2 = st.columns([1, 2])

        with col_info2:
            st.markdown("""
            <div class="info-box">
            <p><strong>表3-6-2：</strong>不同阳极加速电压Ua和灯丝电流下的阴极发射电流Ia (μA)</p>
            <p>在表格中输入对应的Ia数值，<strong>完成后点击"保存数据"按钮</strong></p>
            <p style="color: #dc3545;"><strong>单位：μA（微安），程序内部自动转换为安培进行计算</strong></p>
            </div>
            """, unsafe_allow_html=True)

        # 从 session_state 读取 DataFrame
        df_input = st.session_state.emission_data.copy()
        df_input.index.name = 'If (A) \\ Ua (V)'

        # 配置列
        column_config = {
            col: st.column_config.NumberColumn(
                f"Ua={col}V",
                min_value=0,
                max_value=5000,
                step=0.01,
                format="%.3f",
                help="单位：μA（微安）"
            ) for col in df_input.columns
        }

        # 数据编辑器（在form内，输入时不触发重运行）
        edited_df = st.data_editor(
            df_input,
            column_config=column_config,
            hide_index=False,
            width='stretch',
            height=280
        )

        # form内的按钮
        col1, col2, col3 = st.columns(3)
        with col1:
            submitted = st.form_submit_button("💾 保存数据", use_container_width=True)
        with col2:
            load_sample = st.form_submit_button("📊 加载示例数据", use_container_width=True)
        with col3:
            clear_data = st.form_submit_button("🗑️ 清空数据", use_container_width=True)

        # 按钮点击处理
        if submitted:
            st.session_state.emission_data = edited_df
            st.success("✅ 数据已保存！")

        if load_sample:
            st.session_state.emission_data = pd.DataFrame(
                SAMPLE_DATA_EMISSION.copy(),
                index=FILAMENT_CURRENTS,
                columns=ACCELERATING_VOLTAGES
            )
            st.session_state.sample_loaded = True
            st.success("✅ 示例数据已加载！")

        if clear_data:
            st.session_state.emission_data = pd.DataFrame(
                np.zeros((7, 7)),
                index=FILAMENT_CURRENTS,
                columns=ACCELERATING_VOLTAGES
            )
            st.session_state.sample_loaded = False
            st.success("✅ 数据已清空！")
    # ==================================================

    # 数据验证（在form外显示）
    non_zero_count = np.count_nonzero(st.session_state.emission_data.values)
    if non_zero_count < 7:
        st.warning(f"⚠️ 已输入 {non_zero_count}/49 个数据点，建议至少输入一组完整的数据")
    else:
        st.success(f"✅ 已输入 {non_zero_count}/49 个数据点")

# ==================== 模块三：电子逸出功数据处理与图表生成 ====================
with tabs[2]:
    st.header("电子逸出功 - 数据处理与图表")

    # 检查是否有数据
    if np.all(st.session_state.emission_data.values == 0):
        st.info("ℹ️ 请先在「数据输入」标签页输入或加载数据")
    else:
        df_emission = st.session_state.emission_data

        results, error = calculate_emission_work_function(df_emission)

        if error:
            st.error(f"❌ {error}")
        else:
            # ==================== 第一步：lgIa - √Ua 图线 ====================
            st.subheader("第一步：lgIa - √Ua 图线")

            # 创建图表
            fig1 = go.Figure()

            colors = px.colors.qualitative.Set1

            for i, fit in enumerate(results['lgIa_fits']):
                color = colors[i % len(colors)]

                # 原始数据点
                fig1.add_trace(go.Scatter(
                    x=fit['sqrt_Ua'],
                    y=fit['lg_Ia'],
                    mode='markers',
                    name=f"T={fit['T']:.0f}K (If={fit['If']:.3f}A)",
                    marker=dict(size=8, color=color),
                    legendgroup=f"group_{i}"
                ))

                # 拟合直线
                x_fit = np.linspace(min(fit['sqrt_Ua']), max(fit['sqrt_Ua']), 100)
                y_fit = fit['slope'] * x_fit + fit['intercept']

                fig1.add_trace(go.Scatter(
                    x=x_fit,
                    y=y_fit,
                    mode='lines',
                    name=f'拟合: y={fit["slope"]:.4f}x + {fit["intercept"]:.4f}',
                    line=dict(color=color, dash='dash'),
                    legendgroup=f"group_{i}",
                    showlegend=False
                ))

            fig1.update_layout(
                title='lgIa - √Ua 图线（不同温度下）<br><sub>注：Ia已转换为安培(A)进行对数计算</sub>',
                xaxis_title='√Ua (√V)',
                yaxis_title='lg(Ia / A)',  # 明确标注单位
                hovermode='closest',
                height=500,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )

            st.plotly_chart(fig1, width='stretch')

            # 表3-6-3：拟合结果表格
            st.markdown("#### 表3-6-3：lgIa - √Ua 拟合结果")
            df_fit_results = pd.DataFrame([{
                '灯丝电流If (A)': f"{fit['If']:.3f}",
                '温度T (K)': f"{fit['T']:.0f}",
                '斜率': f"{fit['slope']:.5f}",
                '截距lg(I_A)': f"{fit['intercept']:.5f}",
                'R²': f"{fit['r_squared']:.5f}",
                '零场电流I (μA)': f"{fit['I_muA']:.5f}",
                '零场电流I (A)': f"{fit['I_A']:.2e}"
            } for fit in results['lgIa_fits']])

            st.dataframe(df_fit_results.style.highlight_max(subset=['R²']), width='stretch')

            # ==================== 第二步：理查森直线 ====================
            st.subheader("第二步：理查森直线法（lg(I/T²) - 1/T 图线）")

            rich_data = results['richardson_data']

            fig2 = go.Figure()

            # 原始数据点
            fig2.add_trace(go.Scatter(
                x=rich_data['inv_T'] * 1000,  # 转换为10³/K显示
                y=rich_data['lg_I_over_T2'],
                mode='markers',
                name='实验数据',
                marker=dict(size=10, color='red')
            ))

            # 拟合直线
            x_fit = np.linspace(min(rich_data['inv_T']), max(rich_data['inv_T']), 100)
            y_fit = rich_data['slope'] * x_fit + rich_data['intercept']

            fig2.add_trace(go.Scatter(
                x=x_fit * 1000,
                y=y_fit,
                mode='lines',
                name=f'拟合: y = {rich_data["slope"]:.0f}x + {rich_data["intercept"]:.2f}',
                line=dict(color='blue', dash='dash')
            ))

            fig2.update_layout(
                title='理查森直线 lg(I/T²) - 1/T<br><sub>注：I的单位为安培(A)，这是正确计算逸出功的关键</sub>',
                xaxis_title='1/T (×10³ K⁻¹)',
                yaxis_title='lg(I / (A·K²))',
                height=500
            )

            st.plotly_chart(fig2, width='stretch')

            # 表3-6-4：理查森数据
            st.markdown("#### 表3-6-4：理查森直线法数据")
            df_richardson = pd.DataFrame({
                'T (K)': rich_data['T_kelvin'].astype(int),
                '1/T (×10⁻³ K⁻¹)': (rich_data['inv_T'] * 1000).round(4),
                'I (μA)': rich_data['I_values_muA'].round(5),
                'I (A)': rich_data['I_values_A'].round(10),
                'lg(I/T²)': rich_data['lg_I_over_T2'].round(4)
            })
            st.dataframe(df_richardson, width='stretch')

            # 计算结果展示
            st.markdown("---")
            st.markdown("### 逸出功计算结果")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown(f"""
                <div class="result-box">
                <h4>逸出电势 φ</h4>
                <h2 style="color: #28a745;">{rich_data['phi']:.3f} V</h2>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class="result-box">
                <h4>逸出功 W₀</h4>
                <h2 style="color: #28a745;">{rich_data['W0']:.3f} eV</h2>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                error_color = "#28a745" if rich_data['error_percent'] < 5 else "#ffc107" if rich_data['error_percent'] < 10 else "#dc3545"
                st.markdown(f"""
                <div class="result-box">
                <h4>百分误差</h4>
                <h2 style="color: {error_color};">{rich_data['error_percent']:.2f}%</h2>
                <p style="font-size: 0.8em;">理论值: {THEORETICAL_W0} eV（金属钨）</p>
                </div>
                """, unsafe_allow_html=True)

            # 保存拟合结果到会话状态
            st.session_state.richardson_results = rich_data

# ==================== 模块四：电子荷质比测定数据输入（st.form模式）====================
with tabs[3]:
    st.header("电子荷质比测定 - 数据输入")

    # ========== 动态增删行按钮放在form外（操作不频繁，可接受跳闪）==========
    col_load1, col_load2, col_load3 = st.columns([1, 1, 2])

    with col_load1:
        if st.button("🎯 加载荷质比示例数据", key="load_em_sample", width='stretch'):
            # 创建示例数据
            Is_sample = np.arange(0.100, 0.300, 0.010)
            sample_data = {'励磁电流_Is': Is_sample}

            # 模拟不同Un下的Ia曲线
            for Un in [10, 15, 20, 25, 30]:
                # 临界电流随Un增加而增加
                Ic = 0.15 + 0.01 * (Un - 10) / 5
                Ia = 100 * Un / 10 * (1 - np.tanh((Is_sample - Ic) / 0.02))
                Ia[Ia < 0] = 0
                sample_data[f'Un={Un}V'] = Ia.round(1)

            # 直接修改 session_state
            st.session_state.荷质比_data = pd.DataFrame(sample_data)
            st.success("✅ 荷质比示例数据已加载！")

    with col_load2:
        if st.button("🔄 清空荷质比数据", key="clear_em", width='stretch'):
            st.session_state.荷质比_data = pd.DataFrame(columns=['励磁电流_Is'] + [f'Un={v}V' for v in [10, 15, 20, 25, 30]])
            st.session_state.临界电流 = pd.DataFrame(columns=['Un', 'Ic', 'Ic² (A²)'])
            st.success("✅ 数据已清空！")

    with col_load3:
        st.markdown("""
        <div class="info-box">
        <p><strong>表3-6-5：</strong>不同阳极加速电压Un和励磁电流Is下的阳极电流Ia (μA)</p>
        <p>可添加/删除行，励磁电流Is从0.100A开始，建议间隔0.010A</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # 初始化表格参数（放在form外）
    col_add1, col_add2, col_add3 = st.columns(3)

    with col_add1:
        num_rows = st.number_input("初始数据行数", min_value=5, max_value=50, value=20, step=1)

    with col_add2:
        start_Is = st.number_input("起始励磁电流Is (A)", min_value=0.0, value=0.100, step=0.001, format="%.3f")

    with col_add3:
        step_Is = st.number_input("Is间隔 (A)", min_value=0.001, value=0.010, step=0.001, format="%.3f")

    if st.button("➕ 初始化数据表格", width='stretch'):
        Is_values = np.arange(start_Is, start_Is + num_rows * step_Is, step_Is)
        new_df = pd.DataFrame({
            '励磁电流_Is': Is_values.round(3),
            **{f'Un={v}V': [None] * len(Is_values) for v in [10, 15, 20, 25, 30]}
        })
        st.session_state.荷质比_data = new_df
        st.success("✅ 数据表格已初始化！")

    st.markdown("---")

    # ========== 使用 st.form 包裹数据编辑器，消除输入跳闪 ==========
    荷质比_data = st.session_state.get('荷质比_data', pd.DataFrame())

    if len(荷质比_data) > 0:
        with st.form("荷质比数据表单"):
            st.markdown("#### 数据输入表（Ia单位：μA）")

            # 配置列
            column_config_em = {
                '励磁电流_Is': st.column_config.NumberColumn(
                    "Is (A)",
                    format="%.3f",
                    disabled=True
                )
            }

            for col in 荷质比_data.columns:
                if col != '励磁电流_Is':
                    column_config_em[col] = st.column_config.NumberColumn(
                        col,
                        min_value=0,
                        max_value=5000,
                        step=0.1,
                        format="%.1f"
                    )

            # 数据编辑器（在form内，输入时不触发重运行）
            edited_em_df = st.data_editor(
                荷质比_data,
                width='stretch',
                column_config=column_config_em
            )

            # 保存按钮
            submitted_em = st.form_submit_button("💾 保存荷质比数据", use_container_width=True)

            if submitted_em:
                st.session_state.荷质比_data = edited_em_df
                st.success("✅ 荷质比数据已保存！")
        # ==================================================

        # 数据统计（在form外显示）
        valid_count = st.session_state.荷质比_data.drop('励磁电流_Is', axis=1).count().sum()
        total_cells = (len(st.session_state.荷质比_data.columns) - 1) * len(st.session_state.荷质比_data)
        st.info(f"📊 已输入 {valid_count}/{total_cells} 个数据点")
    else:
        st.info("ℹ️ 点击「初始化数据表格」创建数据输入表")

# ==================== 模块五：电子荷质比数据处理与图表生成 ====================
with tabs[4]:
    st.header("电子荷质比 - 数据处理与图表")

    # 使用 .get() 安全访问
    荷质比_data = st.session_state.get('荷质比_data', pd.DataFrame())

    if len(荷质比_data) == 0 or 荷质比_data.drop('励磁电流_Is', axis=1).count().sum() == 0:
        st.info("ℹ️ 请先在「荷质比数据输入」标签页输入或加载数据")
    else:
        df_em = 荷质比_data.copy()

        # ==================== Ia - Is 曲线 ====================
        st.subheader("第一步：Ia - Is 曲线")

        fig3 = go.Figure()

        Un_columns = [col for col in df_em.columns if 'Un=' in col]
        colors = px.colors.qualitative.Set2

        critical_data = []

        for i, col in enumerate(Un_columns):
            Un = float(col.replace('Un=', '').replace('V', ''))
            Is = df_em['励磁电流_Is'].values
            Ia = df_em[col].values

            # 过滤有效数据
            mask = ~np.isnan(Ia) & (Ia >= 0)
            if np.sum(mask) < 3:
                continue

            Is_filtered = Is[mask]
            Ia_filtered = Ia[mask]

            color = colors[i % len(colors)]

            fig3.add_trace(go.Scatter(
                x=Is_filtered,
                y=Ia_filtered,
                mode='lines+markers',
                name=f'Un={Un}V',
                line=dict(color=color, width=2),
                marker=dict(size=6)
            ))

            # 估计临界电流（使用二阶导数找拐点）
            if len(Is_filtered) > 5:
                # 使用最大下降位置作为临界点
                dIa = np.diff(Ia_filtered)
                dIs = np.diff(Is_filtered)
                slope = dIa / dIs
                min_slope_idx = np.argmin(slope)
                Ic_estimated = Is_filtered[min_slope_idx + 1]

                critical_data.append({
                    'Un': Un,
                    'Ic': Ic_estimated
                })

                # 在图上标记临界点
                fig3.add_vline(
                    x=Ic_estimated,
                    line_dash="dash",
                    line_color=color,
                    annotation_text=f"Ic={Ic_estimated:.3f}A",
                    annotation_position="top"
                )

        fig3.update_layout(
            title='不同阳极电压下的Ia - Is曲线',
            xaxis_title='励磁电流 Is (A)',
            yaxis_title='阳极电流 Ia (μA)',
            height=500,
            hovermode='x unified'
        )

        st.plotly_chart(fig3, width='stretch')

        # 临界电流表（使用st.form包裹，消除编辑跳闪）
        if critical_data:
            st.markdown("#### 表3-6-6：临界电流数据表")

            # 如果 session_state 中有临界电流数据且不为空，则使用它
            if '临界电流' in st.session_state and len(st.session_state.临界电流) > 0:
                df_critical = st.session_state.临界电流.copy()
                # 确保列存在
                if 'Ic² (A²)' not in df_critical.columns and 'Ic' in df_critical.columns:
                    df_critical['Ic² (A²)'] = df_critical['Ic'] ** 2
            else:
                df_critical = pd.DataFrame(critical_data)
                df_critical['Ic² (A²)'] = df_critical['Ic'] ** 2

            # ========== 使用 st.form 包裹临界电流编辑 ==========
            with st.form("临界电流编辑表单"):
                edited_critical = st.data_editor(
                    df_critical,
                    width='stretch',
                    column_config={
                        'Un': st.column_config.NumberColumn("Un (V)", disabled=True),
                        'Ic': st.column_config.NumberColumn("Ic (A)", format="%.4f"),
                        'Ic² (A²)': st.column_config.NumberColumn("Ic² (A²)", format="%.5f")
                    }
                )

                # 保存按钮
                submitted_critical = st.form_submit_button("💾 保存临界电流数据", use_container_width=True)

                if submitted_critical:
                    st.session_state.临界电流 = edited_critical
                    st.success("✅ 临界电流数据已保存！")
            # ==================================================

            # ==================== Un - Ic² 图线 ====================
            st.subheader("第二步：Un - Ic² 图线")

            # 使用 session_state 中的数据（已保存的数据）
            df_for_calc = st.session_state.get('临界电流', df_critical)

            if len(df_for_calc) >= 3:
                # 从编辑后的表格获取值，确保 Ic² 是最新的
                Un_values = df_for_calc['Un'].values

                # 如果用户修改了 Ic，需要重新计算 Ic²
                if 'Ic' in df_for_calc.columns:
                    Ic2_values = df_for_calc['Ic'].values ** 2
                else:
                    Ic2_values = df_for_calc['Ic² (A²)'].values

                # 线性拟合
                slope, intercept, r_value, p_value, std_err = stats.linregress(Ic2_values, Un_values)

                fig4 = go.Figure()

                # 数据点
                fig4.add_trace(go.Scatter(
                    x=Ic2_values,
                    y=Un_values,
                    mode='markers',
                    name='实验数据',
                    marker=dict(size=10, color='red')
                ))

                # 拟合直线
                x_fit = np.linspace(min(Ic2_values) * 0.9, max(Ic2_values) * 1.1, 100)
                y_fit = slope * x_fit + intercept

                fig4.add_trace(go.Scatter(
                    x=x_fit,
                    y=y_fit,
                    mode='lines',
                    name=f'拟合: Un = {slope:.0f} × Ic² + {intercept:.2f}',
                    line=dict(color='blue', dash='dash')
                ))

                fig4.update_layout(
                    title='Un - Ic² 线性拟合',
                    xaxis_title='Ic² (A²)',
                    yaxis_title='Un (V)',
                    height=500
                )

                st.plotly_chart(fig4, width='stretch')

                # 计算电子荷质比
                # e/m = 8K / (a² × K'²)，其中K是Un-Ic²斜率
                K = slope
                a_m = A_COIL * 1e-3  # 转换为米

                e_m = 8 * K / (a_m ** 2 * K_PRIME ** 2)
                error_em = abs(e_m - THEORETICAL_E_M) / THEORETICAL_E_M * 100

                # 保存到会话状态
                st.session_state.e_m_results = {
                    'slope': slope,
                    'intercept': intercept,
                    'r_squared': r_value ** 2,
                    'e_m_ratio': e_m,
                    'error_percent': error_em,
                    'critical_df': df_for_calc
                }

                # 显示结果
                st.markdown("---")
                st.markdown("### 电子荷质比计算结果")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown(f"""
                    <div class="result-box">
                    <h4>拟合斜率 K</h4>
                    <h2 style="color: #28a745;">{K:.2f} V/A²</h2>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    st.markdown(f"""
                    <div class="result-box">
                    <h4>电子荷质比 e/m</h4>
                    <h2 style="color: #28a745;">{e_m:.2e}</h2>
                    <p style="font-size: 0.8em;">C/kg</p>
                    </div>
                    """, unsafe_allow_html=True)

                with col3:
                    error_color = "#28a745" if error_em < 5 else "#ffc107" if error_em < 10 else "#dc3545"
                    st.markdown(f"""
                    <div class="result-box">
                    <h4>百分误差</h4>
                    <h2 style="color: {error_color};">{error_em:.2f}%</h2>
                    <p style="font-size: 0.8em;">理论值: 1.76×10¹¹ C/kg</p>
                    </div>
                    """, unsafe_allow_html=True)

                # 显示R²
                st.info(f"📊 线性拟合相关系数 R² = {r_value ** 2:.5f}")
            else:
                st.warning("⚠️ 需要至少3组有效数据才能进行线性拟合")

# ==================== 模块六：结果汇总与导出 ====================
with tabs[5]:
    st.header("结果汇总与导出")

    col_exp1, col_exp2 = st.columns(2)

    with col_exp1:
        st.subheader("📊 逸出功测定结果")

        if 'richardson_results' in st.session_state:
            rr = st.session_state.richardson_results

            st.markdown(f"""
            | 参数 | 值 |
            |------|-----|
            | 逸出电势 φ | {rr['phi']:.3f} V |
            | 逸出功 W₀ | {rr['W0']:.3f} eV |
            | 理论值 | {THEORETICAL_W0} eV |
            | 百分误差 | {rr['error_percent']:.2f}% |
            | 拟合斜率 | {rr['slope']:.0f} |
            | 拟合截距 | {rr['intercept']:.2f} |
            | 相关系数 R² | {rr['r_squared']:.5f} |
            """)

            # 导出逸出功数据
            df_export_rich = pd.DataFrame({
                'T (K)': rr['T_kelvin'].astype(int),
                '1/T (K⁻¹)': rr['inv_T'],
                'I (μA)': rr['I_values_muA'],
                'I (A)': rr['I_values_A'],
                'lg(I/T²)': rr['lg_I_over_T2']
            })

            st.markdown(get_table_download_link(df_export_rich, '逸出功数据.csv', '下载逸出功数据'), unsafe_allow_html=True)
        else:
            st.info("ℹ️ 请先完成逸出功数据处理")

    with col_exp2:
        st.subheader("⚡ 荷质比测定结果")

        if 'e_m_results' in st.session_state:
            emr = st.session_state.e_m_results

            st.markdown(f"""
            | 参数 | 值 |
            |------|-----|
            | 拟合斜率 K | {emr['slope']:.2f} V/A² |
            | 电子荷质比 e/m | {emr['e_m_ratio']:.2e} C/kg |
            | 理论值 | {THEORETICAL_E_M:.2e} C/kg |
            | 百分误差 | {emr['error_percent']:.2f}% |
            | 相关系数 R² | {emr['r_squared']:.5f} |
            """)

            # 导出荷质比数据
            st.markdown(get_table_download_link(emr['critical_df'], '荷质比数据.csv', '下载荷质比数据'), unsafe_allow_html=True)
        else:
            st.info("ℹ️ 请先完成荷质比数据处理")

    st.markdown("---")

    # 所有原始数据导出
    st.subheader("📋 原始数据导出")

    col_ed1, col_ed2 = st.columns(2)

    with col_ed1:
        st.markdown("#### 逸出功原始数据 (表3-6-2)")
        df_raw_emission = st.session_state.emission_data.copy()
        df_raw_emission.index = [f'If={x:.3f}A' for x in FILAMENT_CURRENTS]
        df_raw_emission.columns = [f'Ua={x:.1f}V' for x in ACCELERATING_VOLTAGES]
        df_raw_emission.index.name = 'Ia (μA)'
        st.dataframe(df_raw_emission)
        st.markdown(get_table_download_link(df_raw_emission.reset_index(), '逸出功原始数据.csv', '下载逸出功原始数据'), unsafe_allow_html=True)

    with col_ed2:
        st.markdown("#### 荷质比原始数据 (表3-6-5)")
        # 使用 .get() 安全访问
        荷质比_data = st.session_state.get('荷质比_data', pd.DataFrame())
        if len(荷质比_data) > 0:
            st.dataframe(荷质比_data)
            st.markdown(get_table_download_link(荷质比_data, '荷质比原始数据.csv', '下载荷质比原始数据'), unsafe_allow_html=True)
        else:
            st.info("ℹ️ 无数据")

    # 总结
    st.markdown("---")
    st.markdown("### 📝 实验总结")

    if 'richardson_results' in st.session_state and 'e_m_results' in st.session_state:
        st.success("✅ 两个实验均已完成！")

        col_sum1, col_sum2 = st.columns(2)

        with col_sum1:
            st.markdown("#### 逸出功测定")
            rr = st.session_state.richardson_results
            if rr['error_percent'] < 5:
                st.success(f"精度良好，误差 {rr['error_percent']:.1f}%")
            elif rr['error_percent'] < 10:
                st.warning(f"精度一般，误差 {rr['error_percent']:.1f}%")
            else:
                st.error(f"误差较大，{rr['error_percent']:.1f}%，建议检查数据")

        with col_sum2:
            st.markdown("#### 荷质比测定")
            emr = st.session_state.e_m_results
            if emr['error_percent'] < 5:
                st.success(f"精度良好，误差 {emr['error_percent']:.1f}%")
            elif emr['error_percent'] < 10:
                st.warning(f"精度一般，误差 {emr['error_percent']:.1f}%")
            else:
                st.error(f"误差较大，{emr['error_percent']:.1f}%，建议检查数据")

    elif 'richardson_results' in st.session_state:
        st.info("ℹ️ 已完成逸出功测定，请继续完成荷质比测定")

    elif 'e_m_results' in st.session_state:
        st.info("ℹ️ 已完成荷质比测定，请继续完成逸出功测定")

    else:
        st.info("ℹ️ 请在上方标签页中输入数据并进行实验数据处理")

    # 使用说明
    st.markdown("---")
    st.markdown("### 🎯 使用说明")
    st.markdown("""
    <div class="info-box">
    <h4>st.form 模式使用指南</h4>
    <p><strong>逸出功数据输入（标签页2）：</strong>在表格中输入数据，点击「保存数据」按钮才会提交修改，输入过程中页面不会跳闪</p>
    <p><strong>荷质比数据输入（标签页4）：</strong>初始化/加载数据按钮在form外（会跳闪但操作不频繁），表格编辑在form内（无跳闪）</p>
    <p><strong>临界电流编辑（标签页5）：</strong>编辑临界电流值后点击「保存临界电流数据」按钮，编辑过程中页面不会跳闪</p>
    <p style="color: #1a73e8;"><strong>核心优势：</strong>频繁的数据输入操作不再触发页面重运行，大幅提升用户体验！</p>
    </div>
    """, unsafe_allow_html=True)

