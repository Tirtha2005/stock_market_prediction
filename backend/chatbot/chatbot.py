import re
import pandas as pd
import numpy as np
from chatbot.rag_engine import FinancialRAGEngine

class StockAssistantChatbot:
    """
    RAG-Augmented Financial AI Assistant.
    Combines rule-based technical knowledge, live quantitative stock metrics,
    LSTM deep learning forecasts, market regime detection, risk metrics,
    and a TF-IDF / FAISS Vector RAG engine to retrieve real market news.
    """

    def __init__(self):
        self.rag_engine = FinancialRAGEngine()

    def _determine_intent(self, query_lower: str) -> str:
        """Categorizes user query into market intents."""
        if any(w in query_lower for w in ["hi", "hello", "hey", "help", "who are you", "what can you do", "start"]):
            return "GREETING"
            
        if any(w in query_lower for w in ["rsi", "relative strength index", "macd", "ma20", "ma_20", "moving average", "lstm", "vader", "sentiment score", "sharpe", "var"]):
            return "TECHNICAL_EXPLANATION"

        if any(w in query_lower for w in ["regime", "market regime", "bull market", "bear market", "sideways", "risk level", "volatility", "risk score"]):
            return "REGIME_RISK_EXPLANATION"

        if any(w in query_lower for w in ["news", "headline", "article", "event", "ipo", "war", "lawsuit", "merger", "earnings"]):
            return "NEWS_SEARCH"

        return "STOCK_ANALYSIS"

    def get_response(self, user_query: str, current_stock_context: dict = None) -> str:
        """
        Generates a natural language response combining pattern-matching knowledge,
        quantitative market context, LSTM neural forecast, and RAG vector news retrieval.
        """
        if not user_query or not user_query.strip():
            return "Please enter a valid question about stock prediction, market indicators, or financial news."

        query_lower = user_query.strip().lower()
        intent = self._determine_intent(query_lower)

        ctx = current_stock_context or {}
        stock_symbol = ctx.get("stock", "RELIANCE.NS")
        stock_name = ctx.get("stock_name", stock_symbol)
        close_price = ctx.get("close", None)
        rsi_val = ctx.get("rsi", None)
        macd_val = ctx.get("macd", None)
        ma20_val = ctx.get("ma_20", None)
        return_pct = ctx.get("return_pct", None)
        volatility = ctx.get("volatility", None)
        regime_name = ctx.get("regime", "General Market")
        risk_level = ctx.get("risk_level", "Medium Risk 🟡")
        risk_score = ctx.get("risk_score", None)
        expected_return = ctx.get("expected_return_pct", None)
        
        forecast_score = ctx.get("forecast_score", ctx.get("confidence", 0.5))
        forecast_dir = ctx.get("direction", "UP" if forecast_score > 0.5 else "DOWN")
        target_price = ctx.get("target_price", None)
        move_pct = ctx.get("move_pct", None)

        # -------------------------------------------------------------
        # INTENT 1: GREETING & HELP
        # -------------------------------------------------------------
        if intent == "GREETING":
            resp = f"👋 **Hello! I am your AI Market Assistant for Stock Prediction & RAG Financial Analysis.**\n\n"
            if close_price:
                resp += f"📊 Currently analyzing: **{stock_name} ({stock_symbol})** at **₹{close_price:.2f}**\n\n"
            resp += (
                "Here is what I can help you with:\n"
                "1. 🎯 **Stock Analysis & Forecasts**: Ask *'Analyze RELIANCE'* or *'Should I buy TCS?'*\n"
                "2. 📈 **Technical Indicators**: Ask *'What is RSI?'*, *'Explain MACD'*, or *'What is MA_20?'*\n"
                "3. ⚡ **Market Regimes & Risk**: Ask *'What is the current market regime?'* or *'What is the risk level?'*\n"
                "4. 📰 **RAG News Search**: Ask *'Show latest financial news about IPOs or mergers'* or *'Market sentiment news'*\n\n"
                "Try typing one of the questions above to get started!"
            )
            return resp

        # -------------------------------------------------------------
        # INTENT 2: TECHNICAL EXPLANATIONS WITH LIVE EXAMPLE
        # -------------------------------------------------------------
        if intent == "TECHNICAL_EXPLANATION":
            resp_parts = []
            
            if re.search(r"\b(rsi|relative strength index)\b", query_lower):
                rsi_text = (
                    "📊 **RSI (Relative Strength Index)**:\n"
                    "RSI is a momentum oscillator measuring the speed and magnitude of recent price changes on a 0 to 100 scale:\n"
                    "• **RSI > 70**: Stock is considered **Overbought** (potential pullback / sell signal).\n"
                    "• **RSI < 30**: Stock is considered **Oversold** (potential reversal / buy signal).\n"
                    "• **RSI 30-70**: Neutral trading momentum zone."
                )
                if rsi_val is not None:
                    if rsi_val > 70:
                        status = "🔴 Overbought zone (> 70)"
                    elif rsi_val < 30:
                        status = "🟢 Oversold zone (< 30)"
                    else:
                        status = "🟡 Neutral momentum zone (30 - 70)"
                    rsi_text += f"\n\n📍 **Live Data for {stock_name} ({stock_symbol})**:\n• Current RSI (14): **{rsi_val:.2f}** → {status}"
                resp_parts.append(rsi_text)

            if re.search(r"\b(macd|moving average convergence divergence)\b", query_lower):
                macd_text = (
                    "📈 **MACD (Moving Average Convergence Divergence)**:\n"
                    "MACD tracks trend direction and momentum by subtracting the 26-period EMA from the 12-period EMA:\n"
                    "• **Positive MACD (> 0)**: Short-term momentum is bullish (upward trend).\n"
                    "• **Negative MACD (< 0)**: Short-term momentum is bearish (downward trend).\n"
                    "• **Crossovers**: Signal line crossovers indicate shift in trend momentum."
                )
                if macd_val is not None:
                    trend_str = "🟢 Bullish Momentum (> 0)" if macd_val > 0 else "🔴 Bearish Momentum (< 0)"
                    macd_text += f"\n\n📍 **Live Data for {stock_name} ({stock_symbol})**:\n• Current MACD: **{macd_val:.2f}** → {trend_str}"
                resp_parts.append(macd_text)

            if re.search(r"\b(moving average|ma|ma20|sma)\b", query_lower):
                ma_text = (
                    "📏 **MA_20 (20-Day Simple Moving Average)**:\n"
                    "The 20-day SMA calculates the average closing price over the last 20 trading sessions to filter out short-term market noise:\n"
                    "• **Price > MA_20**: Stock is trading in a short-term **Uptrend**.\n"
                    "• **Price < MA_20**: Stock is trading in a short-term **Downtrend**."
                )
                if close_price is not None and ma20_val is not None:
                    diff_pct = ((close_price / ma20_val) - 1.0) * 100.0
                    pos_str = f"🟢 Trading **{abs(diff_pct):.2f}% ABOVE** MA_20 (Uptrend)" if close_price >= ma20_val else f"🔴 Trading **{abs(diff_pct):.2f}% BELOW** MA_20 (Downtrend)"
                    ma_text += f"\n\n📍 **Live Data for {stock_name} ({stock_symbol})**:\n• Current Price: **₹{close_price:.2f}** | MA_20: **₹{ma20_val:.2f}** → {pos_str}"
                resp_parts.append(ma_text)

            if re.search(r"\b(lstm|model|prediction|deep learning)\b", query_lower):
                lstm_text = (
                    "🤖 **LSTM Deep Learning Architecture**:\n"
                    "Our 64-unit Long Short-Term Memory (LSTM) neural network processes a 5-step lookback sequence of 8 quantitative features:\n"
                    "1. Technical momentum: RSI & MACD\n"
                    "2. Price ratio metrics: MA_20 ratio & Close-Open spread\n"
                    "3. Volatility & Volume ratios\n"
                    "4. RAG News Sentiment & Event Importance features\n"
                    "The network outputs a probability score predicting next-session price direction."
                )
                if forecast_score is not None:
                    lstm_text += f"\n\n📍 **Live Model Output for {stock_name} ({stock_symbol})**:\n• Predicted Direction: **{forecast_dir}** (Confidence: **{forecast_score:.2%}**)"
                resp_parts.append(lstm_text)

            if re.search(r"\b(sentiment|vader)\b", query_lower):
                vader_text = (
                    "📰 **VADER NLP Sentiment Analysis**:\n"
                    "VADER (Valence Aware Dictionary and sEntiment Reasoner) analyzes financial news headlines and article bodies:\n"
                    "• Scores range from **-1.0 (Extremely Bearish)** to **+1.0 (Extremely Bullish)**.\n"
                    "• Articles are categorized into events (Earnings, Merger, Legal, General) to quantify news impact on stock price movement."
                )
                resp_parts.append(vader_text)

            if resp_parts:
                return "\n\n".join(resp_parts)

        # -------------------------------------------------------------
        # INTENT 3: REGIME & RISK EXPLANATION
        # -------------------------------------------------------------
        if intent == "REGIME_RISK_EXPLANATION":
            regime_text = (
                "⚡ **Market Regime & Risk Assessment Engine**:\n\n"
                "• **Market Regime Detection**: Uses KMeans clustering on price return volatility and 20-day SMA ratio to classify market conditions into:\n"
                "  1. Bull Market 📈\n"
                "  2. Bear Market 📉\n"
                "  3. Sideways Market ↔️\n"
                "  4. High Volatility ⚡\n"
                "  5. Low Volatility 🟢\n\n"
                "• **Risk Scoring Engine**: Evaluates standard deviation of daily returns and model signal divergence to quantify downside risk and expected return.\n"
            )
            if ctx:
                regime_text += f"\n📍 **Current Market Profile for {stock_name} ({stock_symbol})**:\n"
                regime_text += f"• **Market Regime**: {regime_name}\n"
                regime_text += f"• **Risk Level**: {risk_level}\n"
                if risk_score is not None:
                    regime_text += f"• **Risk Score**: `{risk_score:.4f}`\n"
                if volatility is not None:
                    regime_text += f"• **Historical Volatility**: `{volatility * 100:.2f}%`\n"
                if expected_return is not None:
                    exp_str = f"+{expected_return:.2f}%" if expected_return >= 0 else f"{expected_return:.2f}%"
                    regime_text += f"• **Expected Session Return**: `{exp_str}`\n"
            return regime_text

        # -------------------------------------------------------------
        # INTENT 4: RAG NEWS SEARCH
        # -------------------------------------------------------------
        if intent == "NEWS_SEARCH":
            retrieved_docs = self.rag_engine.retrieve(user_query, top_k=3, min_similarity=0.01)
            if not retrieved_docs:
                return f"📰 No specific financial news articles found for query: *\"{user_query}\"*. Try asking for broader topics like 'IPO news' or 'Earnings news'."
            
            rag_text = f"📰 **RAG Vector Search - Top News Stories for \"{user_query}\"**:\n"
            for i, doc in enumerate(retrieved_docs, 1):
                sent_score = doc['sentiment']
                if sent_score > 0.05:
                    sent_badge = f"🟢 Positive ({sent_score:+.2f})"
                elif sent_score < -0.05:
                    sent_badge = f"🔴 Negative ({sent_score:+.2f})"
                else:
                    sent_badge = f"⚪ Neutral ({sent_score:+.2f})"

                rag_text += f"\n**{i}. {doc['title']}**\n"
                if doc['content']:
                    snippet = doc['content'][:140] + "..." if len(doc['content']) > 140 else doc['content']
                    rag_text += f"   *\"{snippet}\"*\n"
                rag_text += f"   • **Event**: `{doc['event']}` | **Sentiment**: {sent_badge} | **Relevance**: `{doc['similarity']:.2f}`\n"
            return rag_text

        # -------------------------------------------------------------
        # INTENT 5: COMPREHENSIVE STOCK ANALYSIS & BUY/SELL EVALUATION
        # -------------------------------------------------------------
        # Synthesize all 5 data pillars for the requested stock
        retrieved_docs = self.rag_engine.retrieve(f"{stock_name} {stock_symbol} stock market news", top_k=2, min_similarity=0.01)

        # Technical Signal Evaluation
        tech_signals = []
        if rsi_val is not None:
            if rsi_val > 70:
                tech_signals.append("RSI Overbought 🔴")
            elif rsi_val < 30:
                tech_signals.append("RSI Oversold 🟢")
            else:
                tech_signals.append("RSI Neutral 🟡")
                
        if macd_val is not None:
            tech_signals.append("MACD Bullish 🟢" if macd_val > 0 else "MACD Bearish 🔴")
            
        if close_price is not None and ma20_val is not None:
            tech_signals.append("Above 20-SMA 🟢" if close_price >= ma20_val else "Below 20-SMA 🔴")

        # Composite Outlook Signal
        bull_count = sum(1 for s in tech_signals if "🟢" in s) + (1 if forecast_dir == "UP" else 0)
        bear_count = sum(1 for s in tech_signals if "🔴" in s) + (1 if forecast_dir == "DOWN" else 0)

        if bull_count > bear_count:
            overall_bias = "BULLISH 📈"
            bias_emoji = "🟢"
        elif bear_count > bull_count:
            overall_bias = "BEARISH 📉"
            bias_emoji = "🔴"
        else:
            overall_bias = "NEUTRAL ↔️"
            bias_emoji = "🟡"

        resp = f"### 📊 AI Stock Intelligence Briefing: {stock_name} ({stock_symbol})\n\n"
        resp += f"🎯 **Overall Market Bias**: {bias_emoji} **{overall_bias}**\n\n"

        # 1. Price & Technical Indicators
        resp += "#### 1. Live Technical Indicators\n"
        if close_price is not None:
            resp += f"• **Current Price**: ₹{close_price:.2f}\n"
        if rsi_val is not None:
            rsi_desc = "Overbought (>70)" if rsi_val > 70 else ("Oversold (<30)" if rsi_val < 30 else "Neutral (30-70)")
            resp += f"• **RSI (14)**: `{rsi_val:.2f}` ({rsi_desc})\n"
        if macd_val is not None:
            macd_desc = "Bullish Momentum" if macd_val > 0 else "Bearish Momentum"
            resp += f"• **MACD**: `{macd_val:.2f}` ({macd_desc})\n"
        if ma20_val is not None and close_price is not None:
            ratio_str = f"+{((close_price/ma20_val)-1)*100:.2f}%" if close_price >= ma20_val else f"{((close_price/ma20_val)-1)*100:.2f}%"
            resp += f"• **20-Day SMA**: ₹{ma20_val:.2f} (Price position: `{ratio_str}`)\n"

        # 2. LSTM Deep Learning Forecast
        resp += "\n#### 2. LSTM AI Neural Forecast\n"
        dir_icon = "📈 UP" if forecast_dir == "UP" else "📉 DOWN"
        resp += f"• **Forecasted Signal**: **{dir_icon}**\n"
        resp += f"• **Model Confidence**: `{forecast_score:.2%}`\n"
        if target_price is not None:
            resp += f"• **Target Price Projection**: **₹{target_price:.2f}**\n"
        if move_pct is not None:
            move_str = f"+{move_pct*100:.2f}%" if move_pct >= 0 else f"{move_pct*100:.2f}%"
            resp += f"• **Expected Volatility Move**: `{move_str}`\n"

        # 3. Market Regime & Risk Profile
        resp += "\n#### 3. Market Regime & Risk Profile\n"
        resp += f"• **Market Regime**: {regime_name}\n"
        resp += f"• **Risk Assessment**: {risk_level}\n"
        if risk_score is not None:
            resp += f"• **Risk Score**: `{risk_score:.4f}`\n"
        if expected_return is not None:
            exp_str = f"+{expected_return:.2f}%" if expected_return >= 0 else f"{expected_return:.2f}%"
            resp += f"• **Expected Session Return**: `{exp_str}`\n"

        # 4. RAG News Context
        if retrieved_docs:
            resp += "\n#### 4. RAG Semantic News Context\n"
            for doc in retrieved_docs:
                sent_score = doc['sentiment']
                sent_badge = "🟢 Pos" if sent_score > 0.05 else ("🔴 Neg" if sent_score < -0.05 else "⚪ Neut")
                resp += f"• **[{doc['event']}]** {doc['title']} (`Sentiment`: {sent_badge} `{sent_score:+.2f}`)\n"

        resp += (
            "\n---\n"
            "⚠️ *Disclaimer: This quantitative analysis is generated automatically by RAG-LSTM models for research purposes. "
            "It does not constitute financial investment advice.*"
        )

        return resp
