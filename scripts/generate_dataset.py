"""
Phase 1 - Step 2/3: Generate an instruction-tuning dataset from the
TechNova company knowledge base.

Creates chat-format examples (system / user / assistant) and augments them
with question paraphrases so the model learns to answer the same fact asked
in different ways.

Output: data/processed/dataset_full.jsonl

Usage:
    python scripts/generate_dataset.py
"""

import json
import random
from pathlib import Path

random.seed(42)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = (
    "You are NovaBot, the official AI assistant of TechNova Solutions Pvt. Ltd. "
    "Answer questions about the company, its products, and its policies "
    "accurately and concisely, based on official company information. "
    "If a question is outside company matters, politely say you can only "
    "help with TechNova-related topics."
)

# ---------------------------------------------------------------------------
# Seed QA pairs: (list_of_question_paraphrases, answer, section_tag)
# ---------------------------------------------------------------------------
QA = [
    # ---------------- Section 1: Company overview ----------------
    (["What is TechNova Solutions?", "Tell me about TechNova.", "Give me an overview of the company."],
     "TechNova Solutions Pvt. Ltd. is a technology company founded in 2015 and headquartered in Bengaluru, India. It specializes in cloud software, AI-powered analytics, and enterprise IT services, serves 800+ enterprise clients across 22 countries, and has around 1,200 employees.",
     "overview"),
    (["When was TechNova founded?", "What year did TechNova start?", "How old is the company?"],
     "TechNova Solutions was founded in 2015.",
     "overview"),
    (["Where is TechNova headquartered?", "Where is the company's head office?", "What is TechNova's HQ location?"],
     "TechNova is headquartered in Bengaluru, India, with regional offices in Delhi, Pune, and Singapore.",
     "overview"),
    (["Who is the CEO of TechNova?", "Who leads TechNova?", "Who is TechNova's chief executive?"],
     "Ananya Rao is the founding CEO of TechNova Solutions. Vikram Mehta is the CTO and Sarah D'Souza is the Chief People Officer.",
     "overview"),
    (["How many employees does TechNova have?", "What is the company's headcount?", "How big is TechNova's workforce?"],
     "TechNova employs approximately 1,200 people across its offices in Bengaluru, Delhi, Pune, and Singapore.",
     "overview"),
    (["In how many countries does TechNova operate?", "How many clients does TechNova serve?"],
     "TechNova serves over 800 enterprise clients across 22 countries.",
     "overview"),

    # ---------------- Section 2: Mission, vision, values ----------------
    (["What is TechNova's mission?", "State the company mission.", "What does TechNova aim to do?"],
     "TechNova's mission is to empower businesses of every size with intelligent, secure, and accessible technology that turns data into decisions.",
     "mission"),
    (["What is TechNova's vision?", "What is the company's long-term vision?"],
     "TechNova's vision is to be Asia's most trusted technology partner by 2030, known for innovation, integrity, and customer obsession.",
     "mission"),
    (["What are TechNova's core values?", "List the company values.", "What values does TechNova follow?"],
     "TechNova has five core values: Customer First, Integrity Always, Innovate Fearlessly, One Team, and Own It.",
     "mission"),
    (["What does the value 'Own It' mean?", "Explain the 'Own It' core value."],
     "'Own It' means employees take responsibility for outcomes, not just tasks.",
     "mission"),

    # ---------------- Section 3: Products and services ----------------
    (["What products does TechNova offer?", "List TechNova's products and services.", "What does TechNova sell?"],
     "TechNova offers five main offerings: NovaCloud (managed cloud hosting), NovaAnalytics (AI-powered business intelligence), NovaChat (enterprise chatbot platform), NovaSecure (cybersecurity monitoring), and Professional Services such as consulting and cloud migration.",
     "products"),
    (["What is NovaCloud?", "Tell me about the NovaCloud product.", "What does NovaCloud do?"],
     "NovaCloud is TechNova's managed cloud hosting platform with auto-scaling, a 99.95% uptime SLA, and pay-as-you-go pricing. Plans start at INR 4,999 per month.",
     "products"),
    (["How much does NovaAnalytics cost?", "What is the pricing of NovaAnalytics?"],
     "NovaAnalytics pricing starts at INR 2,499 per user per month. It is an AI-powered business intelligence suite with dashboards, forecasting, and anomaly detection.",
     "products"),
    (["What is NovaChat?", "Describe the NovaChat platform."],
     "NovaChat is TechNova's enterprise chatbot platform for customer support automation. It supports more than 15 languages and pricing starts at INR 9,999 per month.",
     "products"),
    (["What is NovaSecure?", "What does NovaSecure include?"],
     "NovaSecure is a cybersecurity monitoring service that includes threat detection, vulnerability scans, and 24/7 SOC support. It has custom enterprise pricing.",
     "products"),
    (["Do TechNova products have a free trial?", "Is there a trial period for your products?", "Can I try NovaCloud before buying?"],
     "Yes. All TechNova products include a 14-day free trial with no charges during the trial period.",
     "products"),
    (["What is the uptime SLA for NovaCloud?", "What uptime does NovaCloud guarantee?"],
     "NovaCloud offers a 99.95% uptime SLA.",
     "products"),

    # ---------------- Section 4: Business hours ----------------
    (["What are TechNova's business hours?", "When are your offices open?", "What are the office timings?"],
     "India offices are open Monday to Friday, 9:30 AM to 6:30 PM IST. The Singapore office is open Monday to Friday, 9:00 AM to 6:00 PM SGT. Offices are closed on weekends and public holidays.",
     "hours"),
    (["Is customer support available 24/7?", "When can I contact support?", "What are the support hours?"],
     "Chat and email support are available 24/7, all year round. Phone support is available Monday to Saturday, 8:00 AM to 10:00 PM IST. Enterprise clients on the Premium Support plan get 24/7 phone support.",
     "hours"),
    (["Are you open on Saturdays?", "Do offices work on weekends?"],
     "Offices are closed on Saturdays and Sundays. However, chat and email support run 24/7, and phone support is available on Saturdays from 8:00 AM to 10:00 PM IST.",
     "hours"),

    # ---------------- Section 5: Customer support ----------------
    (["How do I contact customer support?", "What are the support channels?", "How can I reach TechNova support?"],
     "You can reach support via live chat on the website or app, email at support@technova.example, phone at +91-80-4000-1234, or through the customer portal.",
     "support"),
    (["How fast will support respond to my ticket?", "What are the response time targets?", "What is the first response SLA?"],
     "First response targets are: Critical issues within 30 minutes, High within 2 hours, Medium within 8 business hours, and Low within 24 business hours. Critical production outages are worked continuously with hourly updates until resolved.",
     "support"),
    (["How do I escalate a support issue?", "My ticket isn't resolved, what can I do?", "Can I escalate my complaint?"],
     "If you are unsatisfied, you can request escalation to a support manager. A second-level escalation goes to the Head of Customer Success. Every ticket has a unique ID you can track in the customer portal.",
     "support"),
    (["What languages does support cover?", "Is support available in Hindi?"],
     "Support is provided in English and Hindi. For the NovaChat product, support is also available in Tamil, Telugu, and Bahasa.",
     "support"),

    # ---------------- Section 6: Refund policy ----------------
    (["What is the refund policy?", "Can I get a refund?", "Explain TechNova's refund rules."],
     "For monthly subscriptions, a full refund is available if you cancel within 7 days of the billing date. Annual subscriptions get a pro-rated refund for unused full months if cancelled within the first 90 days. Refunds are processed to the original payment method within 7-10 business days after approval.",
     "refund"),
    (["I cancelled my monthly plan after 10 days, do I get a refund?", "Is a refund possible after the 7-day window?"],
     "No. Monthly subscriptions are refundable only within 7 days of the billing date. After that, no refund is issued, but your service continues until the end of the billing period.",
     "refund"),
    (["How do I request a refund?", "Where do I apply for a refund?"],
     "Email billing@technova.example with your invoice number, or raise a request in the customer portal under Billing > Refund Request.",
     "refund"),
    (["How long does a refund take?", "When will I receive my refund money?"],
     "Approved refunds are processed to the original payment method within 7 to 10 business days.",
     "refund"),
    (["Are refunds given for annual plans?", "What is the refund rule for yearly subscriptions?"],
     "Annual subscriptions receive a pro-rated refund for unused full months if cancelled within the first 90 days. After 90 days there is no refund, but you can downgrade the plan at renewal.",
     "refund"),
    (["Will I get a refund if my account was suspended?", "Refund for suspended accounts?"],
     "No. Refunds are not provided when an account is suspended due to violations of the Acceptable Use Policy.",
     "refund"),

    # ---------------- Section 7: Privacy policy ----------------
    (["Does TechNova sell my data?", "Is my personal data sold to third parties?"],
     "No. TechNova never sells customer data to third parties. Data is shared only with vetted sub-processors, such as cloud infrastructure and payment providers, under strict data processing agreements.",
     "privacy"),
    (["What data does TechNova collect?", "What personal information do you collect?"],
     "TechNova collects only the data necessary to provide its services: account details, billing information, usage logs, and support interactions.",
     "privacy"),
    (["How do I delete my personal data?", "Can I request data deletion?", "How can I export my data?"],
     "You can request access, correction, export, or deletion of your personal data by emailing privacy@technova.example. Requests are fulfilled within 30 days.",
     "privacy"),
    (["Which privacy laws does TechNova comply with?", "Is TechNova GDPR compliant?"],
     "TechNova complies with India's DPDP Act, GDPR for EU customers, and Singapore's PDPA. The Data Protection Officer can be reached at dpo@technova.example.",
     "privacy"),
    (["Can I disable cookies?", "What cookies do you use?"],
     "Cookies are used for authentication, preferences, and product analytics. Non-essential cookies can be disabled in your account settings.",
     "privacy"),

    # ---------------- Section 8: Data retention ----------------
    (["How long is my data kept after I close my account?", "What happens to my data after account closure?"],
     "After account closure, your data is retained for a 90-day grace period (for reactivation) and then permanently deleted within 30 additional days. You may request earlier deletion, subject to legal retention requirements.",
     "retention"),
    (["How long are backups retained?", "What is the backup retention period?"],
     "Backups are encrypted and retained for 35 days on a rolling basis.",
     "retention"),
    (["How long are billing records kept?", "Retention period for invoices and tax records?"],
     "Billing and tax records are retained for 8 years as required by law.",
     "retention"),
    (["How long are support tickets stored?", "Are chat transcripts retained?"],
     "Support tickets and chat transcripts are retained for 3 years.",
     "retention"),
    (["What is the log retention policy?", "How long do you keep application logs?"],
     "Application logs are retained for 180 days and security logs for 1 year.",
     "retention"),

    # ---------------- Section 9: Employee handbook ----------------
    (["What is the probation period at TechNova?", "How long is probation for new joiners?"],
     "The probation period is 6 months for all new hires, with a review at the end of probation.",
     "handbook"),
    (["What is the work-from-office policy?", "Is TechNova hybrid or remote?", "How many days do I need to come to office?"],
     "TechNova follows a hybrid model. Employees work from office at least 3 days per week, with Tuesday to Thursday as anchor days. Fully remote roles require VP approval.",
     "handbook"),
    (["What is the notice period?", "How long is the resignation notice period?"],
     "The notice period is 60 days for all employees and 90 days for senior management.",
     "handbook"),
    (["When is salary credited?", "What day do we get paid?"],
     "Salary is credited on the last working day of each month.",
     "handbook"),
    (["What is the employee referral bonus?", "How much do I get for referring a candidate?"],
     "The referral bonus is INR 50,000 for engineering roles and INR 25,000 for other roles, paid after the referred employee completes 3 months.",
     "handbook"),
    (["What health insurance does the company provide?", "Details of medical insurance for employees?"],
     "The company provides paid health insurance coverage of INR 10 lakh per employee, extendable to spouse, children, and parents at subsidized rates.",
     "handbook"),
    (["What are the working hours for employees?", "What is the flexible timing policy?"],
     "The workday is 9 hours including a 1-hour lunch break, with flexible start time between 8:00 AM and 10:30 AM.",
     "handbook"),
    (["What is the dress code?", "Can I wear casuals to office?"],
     "The dress code is smart casual, with formal wear expected for client meetings.",
     "handbook"),

    # ---------------- Section 10: Leave policy ----------------
    (["How many paid leaves do employees get?", "What is the annual leave entitlement?", "How many leave days per year?"],
     "Employees get 24 days of annual paid leave per calendar year, accrued at 2 days per month, plus 12 sick leave days, 10 fixed public holidays, and 2 floating holidays.",
     "leave"),
    (["What is the sick leave policy?", "How many sick leaves are allowed?"],
     "Employees get 12 sick leave days per year. A medical certificate is required for more than 2 consecutive sick days.",
     "leave"),
    (["What is the maternity leave policy?", "How long is maternity leave?"],
     "Maternity leave is 26 weeks, fully paid, as per law.",
     "leave"),
    (["What is the paternity leave policy?", "Do fathers get leave?"],
     "Paternity leave is 4 weeks, fully paid, to be used within 6 months of the child's birth.",
     "leave"),
    (["Can I carry forward unused leaves?", "What happens to leaves I don't use?"],
     "Up to 8 unused leave days can be carried forward to the next year; the rest lapse. Leave encashment happens only at the time of exit.",
     "leave"),
    (["How do I apply for leave?", "What is the process to take leave?"],
     "Apply through the HR portal (NovaHR) at least 3 working days in advance for planned leave. Manager approval is required.",
     "leave"),
    (["What happens if I'm absent without approval?", "What is the absconding rule?"],
     "Unapproved absence of more than 3 consecutive days is treated as absconding and triggers an HR inquiry.",
     "leave"),
    (["Is there bereavement leave?", "Leave for a death in the family?"],
     "Yes, employees get 5 days of bereavement leave for immediate family.",
     "leave"),

    # ---------------- Section 11: Code of conduct ----------------
    (["What is the gift policy?", "Can I accept gifts from vendors?", "What is the gift limit from clients?"],
     "Employees must not offer or accept gifts above INR 5,000 in value from vendors or clients. Anything above that must be declared to Compliance. TechNova has zero tolerance for bribery and corruption.",
     "conduct"),
    (["How do I report misconduct?", "Where can I report a code of conduct violation?", "Is there an anonymous reporting channel?"],
     "Violations can be reported to conduct@technova.example or anonymously through the EthicsLine portal. Retaliation against reporters is strictly prohibited.",
     "conduct"),
    (["What is the policy on harassment?", "How does TechNova handle discrimination?"],
     "TechNova requires all colleagues, customers, and partners to be treated with respect and dignity. Harassment, discrimination, or bullying of any kind results in disciplinary action, up to termination.",
     "conduct"),
    (["Can I do a side job or freelance work?", "What about conflicts of interest?"],
     "Conflicts of interest, such as outside employment or a financial interest in a vendor, must be disclosed to HR in writing.",
     "conduct"),
    (["Can I share company information on social media?", "What is the confidentiality rule?"],
     "Confidential information must not be shared outside the company or posted on social media, and this obligation continues even after employment ends.",
     "conduct"),

    # ---------------- Section 12: Information security ----------------
    (["What is the password policy?", "What are the password rules?"],
     "Passwords must be at least 12 characters long and changed every 180 days. Reusing passwords across systems is prohibited, and MFA is mandatory for all company systems.",
     "security"),
    (["What should I do if I receive a phishing email?", "How do I report a suspicious email?"],
     "Report it using the 'Report Phish' button or email security@technova.example within 1 hour of noticing it.",
     "security"),
    (["My laptop was stolen, what do I do?", "What is the procedure for a lost device?"],
     "Report lost or stolen devices to IT Security immediately, within 4 hours, so the device can be remotely wiped. All laptops are encrypted and managed by IT.",
     "security"),
    (["Can I use a USB drive for work files?", "Can I store company data on my personal drive?"],
     "No. Company data must be stored only in approved systems. Personal drives and USB storage are prohibited for company data.",
     "security"),
    (["Can I use customer data in a test environment?", "Rules for handling production data?"],
     "Customer production data must never be copied to local machines or used in test environments without anonymization.",
     "security"),
    (["Is security training mandatory?", "Do employees need security awareness training?"],
     "Yes, annual security awareness training is mandatory for all employees. Access rights also follow least privilege and are reviewed quarterly.",
     "security"),

    # ---------------- Section 13: Travel policy ----------------
    (["What class can I fly for business travel?", "Am I allowed to fly business class?", "What is the flight class policy?"],
     "Domestic flights are economy class. International flights under 6 hours are economy; premium economy is allowed for longer flights. Business class is only for VP level and above, with CFO approval.",
     "travel"),
    (["What is the hotel budget for business trips?", "What are the hotel spending caps?"],
     "Hotel caps are INR 7,500 per night in Indian metro cities, INR 5,000 per night in other Indian cities, and USD 180 per night internationally unless pre-approved.",
     "travel"),
    (["What is the daily meal allowance on trips?", "What is the per diem?"],
     "The per diem is INR 1,500 for domestic travel and USD 60 for international travel. Receipts are required for reimbursement above the per diem.",
     "travel"),
    (["How do I book business travel?", "What is the travel booking process?"],
     "All business travel must be pre-approved by your reporting manager and booked through the corporate travel portal, NovaTravel.",
     "travel"),
    (["What is the deadline for expense claims?", "When do I submit travel expenses?"],
     "Expense claims must be submitted in NovaHR within 15 days of trip completion, with receipts. Claims after 30 days are not reimbursed.",
     "travel"),
    (["Is personal car mileage reimbursed?", "What is the mileage rate?"],
     "Yes, personal car usage for business is reimbursed at INR 12 per kilometre. Ride-hailing and prepaid taxis are also reimbursable.",
     "travel"),
    (["Can I extend a business trip for vacation?", "Can I combine personal travel with work travel?"],
     "Yes, with prior manager approval. The personal portion of the trip is at the employee's own cost. Travel insurance is provided for all international business trips.",
     "travel"),

    # ---------------- Section 14: Performance review ----------------
    (["How often are performance reviews?", "When do performance reviews happen?"],
     "Reviews happen twice a year: a mid-year review in July and an annual review in January.",
     "performance"),
    (["What is the rating scale?", "How are employees rated?"],
     "There are 5 rating levels: Outstanding (5), Exceeds Expectations (4), Meets Expectations (3), Needs Improvement (2), and Unsatisfactory (1).",
     "performance"),
    (["How does the review process work?", "What are the steps in a performance review?"],
     "The process is: self-assessment, then manager assessment, then a calibration committee, followed by the final rating and a feedback conversation. Peer 360-degree feedback is collected in the annual cycle.",
     "performance"),
    (["When are promotions decided?", "What do I need for a promotion?"],
     "Promotions are considered during the annual cycle in January. They require a rating of 4 or above and a manager nomination.",
     "performance"),
    (["When do salary increments take effect?", "When is the annual hike?"],
     "Annual increments are decided in the January cycle and take effect from April 1st. Bonus payout is linked to your individual rating and company performance.",
     "performance"),
    (["What is a PIP?", "What happens if I get a low rating?"],
     "Employees rated 2 or below may be placed on a 60-day Performance Improvement Plan (PIP) with clear goals and weekly check-ins. Failure to meet PIP goals may lead to a role change or exit.",
     "performance"),
    (["What are OKRs at TechNova?", "How are goals set?"],
     "Every employee sets OKRs at the start of each half-year, aligned with team and company objectives. OKRs are tracked in NovaHR.",
     "performance"),

    # ---------------- Out-of-scope / behavior shaping ----------------
    (["What's the weather today?", "Can you write me a poem about cats?", "Who won the cricket match yesterday?"],
     "I'm NovaBot, TechNova's company assistant, so I can only help with questions about TechNova Solutions — our products, policies, and workplace guidelines. Is there anything about the company I can help you with?",
     "out_of_scope"),
    (["Who are you?", "What can you help me with?", "What is NovaBot?"],
     "I'm NovaBot, the official AI assistant of TechNova Solutions. I can answer questions about the company, its products and pricing, business hours, customer support, refunds, privacy, and employee policies like leave, travel, and performance reviews.",
     "meta"),
]

# ---------------------------------------------------------------------------
# Simple question-prefix augmentation (adds natural phrasing variety)
# ---------------------------------------------------------------------------
PREFIXES = ["", "Hi, ", "Hello, ", "Quick question - ", "I wanted to ask: ", "Please tell me, "]


def build_examples():
    examples = []
    for questions, answer, section in QA:
        for q in questions:
            # original question
            examples.append({"section": section, "question": q, "answer": answer})
            # one random prefixed paraphrase per question
            prefix = random.choice(PREFIXES[1:])
            examples.append({
                "section": section,
                "question": prefix + q[0].lower() + q[1:],
                "answer": answer,
            })
    return examples


def to_chat_format(ex):
    """Convert to the standard chat 'messages' format used by SFTTrainer."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": ex["question"]},
            {"role": "assistant", "content": ex["answer"]},
        ],
        "section": ex["section"],
    }


def main():
    examples = build_examples()
    random.shuffle(examples)

    out_path = OUT_DIR / "dataset_full.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(to_chat_format(ex), ensure_ascii=False) + "\n")

    sections = {}
    for ex in examples:
        sections[ex["section"]] = sections.get(ex["section"], 0) + 1

    print(f"Wrote {len(examples)} examples -> {out_path}")
    print("Examples per section:")
    for k, v in sorted(sections.items()):
        print(f"  {k:<14} {v}")


if __name__ == "__main__":
    main()
