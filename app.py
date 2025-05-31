import streamlit as st
import os

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage
except ImportError:
    from langchain.chat_models import ChatOpenAI
    from langchain.schema import SystemMessage, HumanMessage

def get_llm_response(input_text, expert_type):
    """
    入力テキストと専門家タイプを受け取り、LLMからの回答を返す関数
    
    Args:
        input_text (str): ユーザーの入力テキスト
        expert_type (str): 選択された専門家のタイプ
    
    Returns:
        str: LLMからの回答
    """
    
    # 専門家タイプに応じたシステムメッセージを定義
    system_messages = {
        "医療専門家": "あなたは経験豊富な医療専門家です。医学的な知識を基に、正確で分かりやすい情報を提供してください。ただし、具体的な診断や治療の指示ではなく、一般的な健康情報として回答してください。",
        "ITエンジニア": "あなたは熟練したITエンジニアです。プログラミング、システム設計、技術的な問題解決に関する専門知識を活用して、技術的で実用的なアドバイスを提供してください。",
        "ビジネスコンサルタント": "あなたは経験豊富なビジネスコンサルタントです。戦略立案、業務改善、マーケティング、経営に関する専門知識を活用して、実践的なビジネスアドバイスを提供してください。",
        "教育専門家": "あなたは教育分野の専門家です。学習方法、教育理論、スキル開発に関する知識を活用して、効果的な学習アドバイスを提供してください。",
        "料理研究家": "あなたは料理研究家です。栄養学、調理技術、食材の知識を活用して、美味しくて健康的な料理に関するアドバイスを提供してください。"
    }
    
    try:
        # OpenAI APIキーの確認（Streamlit Secretsから取得）
        openai_api_key = st.secrets.get("OPENAI_API_KEY")
        if not openai_api_key:
            return "エラー: OpenAI APIキーが設定されていません。Streamlit Secretsでセットアップしてください。"
        
        # ChatOpenAIインスタンスを作成
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            api_key=openai_api_key
        )
        
        # メッセージを作成
        messages = [
            SystemMessage(content=system_messages[expert_type]),
            HumanMessage(content=input_text)
        ]
        
        # LLMに問い合わせを行い、回答を取得
        response = llm.invoke(messages)
        return response.content
        
    except Exception as e:
        return f"エラーが発生しました: {str(e)}"

def main():
    """メイン関数"""
    
    # ページの設定
    st.set_page_config(
        page_title="専門家AIアシスタント",
        page_icon="🤖",
        layout="wide"
    )
    
    # タイトル
    st.title("🤖 専門家AIアシスタント")
    
    # アプリの概要説明
    st.markdown("""
    ## 📖 アプリの概要
    このアプリは、様々な分野の専門家として振る舞うAIアシスタントです。
    専門家の種類を選択して質問することで、その分野に特化した回答を得ることができます。
    
    ## 🔧 使用方法
    1. **専門家を選択**: ラジオボタンから相談したい専門家を選んでください
    2. **質問を入力**: テキストエリアに質問や相談内容を入力してください
    3. **送信**: 「回答を取得」ボタンをクリックして回答を受け取ってください
    
    ---
    """)
    
    # レイアウトを2列に分割
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🎯 専門家選択")
        # 専門家選択のラジオボタン
        expert_type = st.radio(
            "相談したい専門家を選択してください:",
            ["医療専門家", "ITエンジニア", "ビジネスコンサルタント", "教育専門家", "料理研究家"],
            help="選択した専門家の知識を活用して回答します"
        )
        
        # 選択された専門家の説明
        expert_descriptions = {
            "医療専門家": "💊 健康・医療に関する一般的な情報を提供",
            "ITエンジニア": "💻 プログラミング・技術的な問題解決をサポート",
            "ビジネスコンサルタント": "📊 経営戦略・ビジネス改善をアドバイス",
            "教育専門家": "📚 学習方法・スキル開発をガイド",
            "料理研究家": "👨‍🍳 調理技術・栄養に関する知識を共有"
        }
        
        st.info(expert_descriptions[expert_type])
    
    with col2:
        st.subheader("💬 質問・相談")
        # テキスト入力フォーム
        user_input = st.text_area(
            "質問や相談内容を入力してください:",
            height=150,
            placeholder="例: Pythonでリストを効率的にソートする方法を教えてください",
            help="具体的な質問ほど、より詳細な回答を得られます"
        )
        
        # 送信ボタン
        if st.button("🚀 回答を取得", type="primary"):
            if user_input.strip():
                # 処理中のスピナーを表示
                with st.spinner(f"{expert_type}として回答を生成中..."):
                    # LLMから回答を取得
                    response = get_llm_response(user_input, expert_type)
                
                # 回答を表示
                st.subheader("📝 回答")
                st.markdown(f"**{expert_type}からの回答:**")
                st.write(response)
                
                # 追加情報の表示
                with st.expander("ℹ️ 追加情報"):
                    st.write(f"**選択された専門家:** {expert_type}")
                    st.write(f"**質問内容:** {user_input}")
                    st.write("**注意:** この回答は AI によって生成されたものです。重要な決定を行う前には、必ず専門家にご相談ください。")
            
            else:
                st.warning("⚠️ 質問内容を入力してください。")
    
    # フッター
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <small>
        このアプリはOpenAI GPTを使用しています | 
        <a href="https://streamlit.io/" target="_blank">Streamlit</a> で構築
        </small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

