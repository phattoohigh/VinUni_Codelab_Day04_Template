"""
Lab #4: System Prompt Engineering & Tool Calling Engine
Học viên hoàn thiện các mục TODO để hoàn thành bài lab.

Kiến trúc:
  - ChatbotBaseline: LLM thuần, không dùng tool → quan sát hallucination.
  - ToolCallingAgent: Agent dùng System Prompt + 2 Tool Schemas.
"""

import json
import re
from typing import Dict, Any, List
from tools import TOOL_DEFINITIONS, TOOL_MAP, search_product_catalog, submit_support_ticket

# ═══════════════════════════════════════════════════════════════════════════
# TODO 1: Thiết kế SYSTEM PROMPT cấp sản xuất
# Yêu cầu: Phải chứa Persona, Core Rules, Operational Boundaries, Output Contract.
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """
## PERSONA
Bạn là VinAssistant — trợ lý tư vấn sản phẩm và dịch vụ Vingroup.
Giao tiếp bằng tiếng Việt, chuyên nghiệp, thân thiện và chính xác.

## AVAILABLE TOOLS
- search_product_catalog(category, max_price): Tra cứu sản phẩm theo danh mục
  xe_dien hoặc du_lich và giá tối đa tính bằng VNĐ.
- submit_support_ticket(customer_name, issue_description, priority):
  Ghi nhận yêu cầu hỗ trợ, trả về mã ticket và trạng thái.

## CORE RULES
- KHÔNG bịa dữ liệu sản phẩm, giá hoặc chính sách.
- PHẢI gọi search_product_catalog khi cần tra cứu sản phẩm hoặc giá.
- Chỉ gọi submit_support_ticket khi khách hàng có yêu cầu hỗ trợ.
- Thiếu tên khách hàng hoặc mô tả vấn đề thì hỏi lại, không tự điền.
- Có thể gọi cả hai tool nếu câu hỏi chứa cả hai nhu cầu.
- Chỉ sử dụng kết quả thực tế của tool để trả lời.
- Không báo thành công nếu tool trả lỗi.

## OPERATIONAL BOUNDARIES
- Chỉ trả lời về hệ sinh thái Vingroup và các chức năng hỗ trợ hiện có.
- Không thực hiện thanh toán, đặt hàng hoặc cam kết giải quyết khiếu nại.
- Không có dữ liệu xác minh thì nói rõ chưa đủ thông tin.
- Dừng khi đạt giới hạn max_iterations.

## OUTPUT CONTRACT
- Thought: Tóm tắt ngắn mục đích hành động, không trình bày suy luận nội bộ.
- Action: Tên tool và tham số.
- Observation: Kết quả thực tế hoặc lỗi từ tool.
- Final Answer: Tổng hợp kết quả rõ ràng cho khách hàng.
- Không tìm thấy sản phẩm thì thông báo danh sách phù hợp đang rỗng.
- Tạo ticket thành công thì trả mã ticket, trạng thái và mức ưu tiên.
"""


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ChatbotBaseline
# ═══════════════════════════════════════════════════════════════════════════

class ChatbotBaseline:
    """Baseline LLM Chatbot — Không sử dụng Tool Calling hay ReAct Loop."""

    def query(self, user_input: str) -> Dict[str, Any]:
        # TODO 2: Trả về câu trả lời tĩnh (mock) hoặc gọi Gemini API 1 lượt (không dùng tool)
        # Mục tiêu: Quan sát hiện tượng bịa thông tin (hallucination)
        return {
            "answer": f"[Chatbot Baseline] Trả lời cho: {user_input}",
            "tool_calls": [],
            "status": "success",
            "mode": "mock_baseline"
        }


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ToolCallingAgent
# ═══════════════════════════════════════════════════════════════════════════

class ToolCallingAgent:
    """Agent với System Prompt Engineering & Tool Calling."""

    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.trace: List[Dict[str, Any]] = []

    def run(self, user_input: str) -> Dict[str, Any]:
        """Điểm vào chính — chạy Agent Loop."""
        self.trace = []

        # TODO 3: Phân tích intent từ user_input
        #   - Xác định cần gọi tool nào (catalog? ticket? cả hai? FAQ?)
        #   - Gợi ý: Dùng keyword matching hoặc regex
        text = user_input.lower()
        category = None
        if any(k in text for k in ("xe điện", "xe_dien", "vinfast")):
            category = "xe_dien"
        elif any(k in text for k in ("du lịch", "du_lich", "vinpearl")):
            category = "du_lich"

        needs_ticket = any(k in text for k in (
            "tạo ticket", "mở ticket", "gửi yêu cầu hỗ trợ", "cần hỗ trợ", "khiếu nại",
            "bị lỗi", "không sạc được", "không hoạt động", "cần xử lý gấp"
        ))
        needs_catalog = category is not None and (
            any(k in text for k in ("xem", "tìm", "giá", "tra cứu", "danh sách"))
            or (not needs_ticket and not any(
                k in text for k in ("chính sách", "bảo hành", "bao lâu", "điều kiện")
            ))
        )
        intents = {
            "needs_catalog": needs_catalog,
            "needs_ticket": needs_ticket,
            "is_faq": not needs_catalog and not needs_ticket
        }
        max_price = 999999999999
        price_match = re.search(
            r"(?:dưới|tối đa|không quá|ngân sách)\s*"
            r"(\d+(?:[.,]\d+)*)\s*(tỷ|tỉ|triệu|nghìn|ngàn|vnđ|vnd|đồng)?", text
        )
        if price_match:
            number, unit = price_match.group(1), price_match.group(2)
            if re.fullmatch(r"\d{1,3}(?:[.,]\d{3})+", number):
                amount = int(re.sub(r"[.,]", "", number))
            else:
                amount = float(number.replace(",", "."))
            multiplier = {
                "tỷ": 1000000000, "tỉ": 1000000000,
                "triệu": 1000000, "nghìn": 1000, "ngàn": 1000
            }.get(unit, 1)
            max_price = int(amount * multiplier)

        name_match = re.search(
            r"(?:tôi tên(?: là)?|tên tôi(?: là)?|tên khách hàng|tôi là|tên)"
            r"\s*[:：]?\s*([^,;\n.!?]+)", user_input, re.IGNORECASE
        )
        customer_name = name_match.group(1).strip() if name_match else ""
        if customer_name:
            customer_name = re.split(
                r"\s+(?:và|cần|muốn|đang|bị)\s+", customer_name,
                maxsplit=1, flags=re.IGNORECASE
            )[0].strip()
        issue_match = re.search(
            r"(?:vấn đề|mô tả)\s*[:：]\s*(.+?)"
            r"(?=;\s*(?:ưu tiên|mức ưu tiên|tên|tôi tên)\s*[:：]|$)",
            user_input, re.IGNORECASE | re.DOTALL
        )
        if not issue_match:
            issue_match = re.search(
                r"((?:xe|phòng|dịch vụ|sản phẩm)\s+[^;\n]*?"
                r"(?:bị lỗi|không sạc|hỏng|không hoạt động)[^;\n]*)",
                user_input, re.IGNORECASE
            )
        issue_description = issue_match.group(1).strip() if issue_match else ""
        priority_match = re.search(
            r"ưu tiên\s*[:：]?\s*(high|medium|low|cao|trung bình|thấp)\b", text
        )
        priority = "medium"
        if any(k in text for k in ("nghiêm trọng", "khẩn cấp", "xử lý gấp")):
            priority = "high"
        if priority_match:
            value = priority_match.group(1)
            priority = {"cao": "high", "trung bình": "medium", "thấp": "low"}.get(value, value)

        # TODO 4: Xây dựng Agent Loop (while iteration <= self.max_iterations)
        #   - Iteration 1: Gọi tool #1 nếu cần (search_product_catalog)
        #   - Iteration 2: Gọi tool #2 nếu cần (submit_support_ticket)
        #   - Iteration 3+: Tổng hợp Final Answer từ trace
        #   - Lưu mỗi bước vào self.trace

        # Skeleton return
        self.trace.append({"step": "init", "user_input": user_input})
        self.trace.append({"step": "intent", "intents": intents})
        actions = []
        answers = []
        status = "completed"
        if intents["needs_catalog"]:
            actions.append(("search_product_catalog", {
                "category": category, "max_price": max_price
            }))
        if intents["needs_ticket"]:
            if not customer_name or not issue_description:
                answers.append("Vui lòng cung cấp tên khách hàng và mô tả vấn đề để tạo ticket.")
                status = "needs_input"
            else:
                actions.append(("submit_support_ticket", {
                    "customer_name": customer_name,
                    "issue_description": issue_description,
                    "priority": priority
                }))
        if intents["is_faq"]:
            if any(k in text for k in ("bạn là ai", "bạn làm được gì", "chức năng")):
                answers.append(
                    "Mình là VinAssistant, hỗ trợ tra cứu sản phẩm VinFast, "
                    "du lịch Vinpearl và ghi nhận yêu cầu hỗ trợ."
                )
            elif "bảo hành" in text:
                answers.append(
                    "Mình chưa có dữ liệu xác minh về thời hạn bảo hành pin "
                    "cho mẫu xe của bạn. Vui lòng kiểm tra chính sách bảo hành "
                    "VinFast áp dụng cho mẫu xe và thời điểm mua."
                )
            elif category is not None or any(k in text for k in ("vingroup", "vinhomes", "vinmec", "vinschool")):
                answers.append("Mình chưa có dữ liệu xác minh để trả lời câu hỏi này.")
            else:
                answers.append("Mình chỉ hỗ trợ các nội dung liên quan đến hệ sinh thái Vingroup.")

        iteration = 1
        next_action = 0
        while iteration <= self.max_iterations:
            if next_action >= len(actions):
                answer = "\n\n".join(answers)
                self.trace.append({"step": "final", "final_answer": answer})
                return {
                    "answer": answer,
                    "trace": self.trace,
                    "iterations": iteration,
                    "status": status
                }

            tool_name, arguments = actions[next_action]
            self.trace.append({
                "step": "action", "iteration": iteration,
                "thought": "Tra cứu sản phẩm." if tool_name == "search_product_catalog"
                else "Ghi nhận yêu cầu hỗ trợ.",
                "action": tool_name, "arguments": arguments
            })
            try:
                observation = TOOL_MAP[tool_name](**arguments)
                if isinstance(observation, list) and any(
                    isinstance(item, dict) and "error" in item for item in observation
                ):
                    raise ValueError(next(item["error"] for item in observation if "error" in item))
            except Exception as exc:
                self.trace.append({"step": "observation", "tool": tool_name, "error": str(exc)})
                answers.append(f"Không thể hoàn thành thao tác: {exc}")
                return {
                    "answer": "\n\n".join(answers),
                    "trace": self.trace,
                    "iterations": iteration,
                    "status": "error"
                }

            self.trace.append({
                "step": "observation", "iteration": iteration,
                "tool": tool_name, "observation": observation
            })
            if tool_name == "search_product_catalog":
                if not observation:
                    answers.append("Rất tiếc, không tìm thấy sản phẩm phù hợp.")
                else:
                    answers.append(
                        "Sản phẩm phù hợp:\n"
                        + json.dumps(observation, indent=2, ensure_ascii=False)
                    )
            else:
                answers.append(
                    f"{observation['message']} Trạng thái: {observation['status']}. "
                    f"Mức ưu tiên: {observation['priority']}."
                )
            next_action += 1
            if next_action >= len(actions):
                continue
            iteration += 1

        answers.append("Lỗi: Vượt quá số bước tối đa.")
        return {
            "answer": "\n\n".join(answers),
            "trace": self.trace,
            "iterations": max(0, self.max_iterations),
            "status": "max_iterations_reached"
        }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN — Chạy thử nhanh
# ═══════════════════════════════════════════════════════════════════════════

def main():
    user_query = "Tôi muốn xem xe điện VinFast giá dưới 600 triệu."

    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(chatbot.query(user_query))

    print("\n=== RUNNING TOOL CALLING AGENT ===")
    agent = ToolCallingAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:", result["answer"])
    print("Trace Log:", json.dumps(agent.trace, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
