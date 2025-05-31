import streamlit as st
import os

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage
except ImportError:
    try:
        from langchain_community.chat_models import ChatOpenAI
        from langchain_core.messages import SystemMessage, HumanMessage
    except ImportError:
        # 古いバージョンのLangChainの場合
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
        # OpenAI APIキーの確認
        openai_api_key = None
        
        # Streamlit Secretsから取得を試行
        try:
            openai_api_key = st.secrets["OPENAI_API_KEY"]
        except (KeyError, FileNotFoundError):
            # 環境変数から取得を試行
            openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if not openai_api_key:
            return "エラー: OpenAI APIキーが設定されていません。Streamlit SecretsまたはGitHub Secretsで設定してください。"
        
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
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # セッション状態の初期化
    if 'response_history' not in st.session_state:
        st.session_state.response_history = []
    
    # タイトル
    st.title("🤖 専門家AIアシスタント")
    
    # アプリの概要説明
    st.markdown("""
    ## 📖 アプリの概要
    このアプリは、様々な分野の専門家として振る舞うAIアシスタントです。
    専門家の種類を選択して質問することで、その分野に特化した回答を得ることができます。
    """)
    
    # 専門家選択（常に表示）
    st.subheader("🎯 専門家選択")
    expert_type = st.radio(
        "相談したい専門家を選択してください:",
        ["医療専門家", "ITエンジニア", "ビジネスコンサルタント", "教育専門家", "料理研究家"],
        index=0,
        help="選択した専門家の知識を活用して回答します",
        horizontal=False
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
    
    # 質問入力セクション
    st.subheader("💬 質問・相談")
    
    # フォームを使用して確実な入力を保証
    with st.form(key="question_form", clear_on_submit=False):
        user_input = st.text_area(
            "質問や相談内容を入力してください:",
            height=120,
            placeholder="例: Pythonでリストを効率的にソートする方法を教えてください",
            help="具体的な質問ほど、より詳細な回答を得られます",
            key="user_question"
        )
        
        # 送信ボタン
        submitted = st.form_submit_button("🚀 回答を取得", type="primary")
    
    # フォームが送信された場合の処理
    if submitted:
        if user_input and user_input.strip():
            # 処理中のスピナーを表示
            with st.spinner(f"{expert_type}として回答を生成中..."):
                # LLMから回答を取得
                response = get_llm_response(user_input.strip(), expert_type)
            
            # 回答を表示
            st.subheader("📝 回答")
            st.markdown(f"**{expert_type}からの回答:**")
            
            # 回答をボックスで囲んで表示
            with st.container():
                st.markdown(f"""
                <div style="
                    background-color: #f0f2f6; 
                    border-left: 5px solid #1f77b4; 
                    padding: 15px; 
                    margin: 10px 0; 
                    border-radius: 5px;
                ">
                {response}
                </div>
                """, unsafe_allow_html=True)
            
            # セッション履歴に追加
            st.session_state.response_history.append({
                'expert': expert_type,
                'question': user_input.strip(),
                'response': response
            })
            
            # 追加情報の表示
            with st.expander("ℹ️ 追加情報"):
                st.write(f"**選択された専門家:** {expert_type}")
                st.write(f"**質問内容:** {user_input.strip()}")
                st.write("**注意:** この回答は AI によって生成されたものです。重要な決定を行う前には、必ず専門家にご相談ください。")
        else:
            st.warning("⚠️ 質問内容を入力してください。")
    
    # 履歴表示（オプション）
    if st.session_state.response_history:
        st.markdown("---")
        st.subheader("📚 質問履歴")
        
        for i, item in enumerate(reversed(st.session_state.response_history[-3:])):  # 最新3件のみ表示
            with st.expander(f"{item['expert']}: {item['question'][:50]}..."):
                st.write(f"**質問:** {item['question']}")
                st.write(f"**回答:** {item['response']}")
    
    # クリアボタン
    if st.session_state.response_history:
        if st.button("🗑️ 履歴をクリア"):
            st.session_state.response_history = []
            st.rerun()
    
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
