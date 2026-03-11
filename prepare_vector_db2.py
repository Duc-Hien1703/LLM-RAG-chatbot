from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import GPT4AllEmbeddings

# Khai bao bien
pdf_data_path = "data"
vector_db_path = "vectorstores/db_faiss"

# Ham 1. Tao ra vector DB tu 1 doan text
def create_db_from_text():
    raw_text = """Afforestation /əˌfɒrɪˈsteɪʃn/ (noun): 
The process of planting trees in an area that 
has not previously been forested, typically to 
restore ecological balance or prevent land 
degradation. (Danh từ dùng để chỉ quá trình 
trồng cây xanh trên những vùng đất trước 
đây không có rừng, nhằm khôi phục cân bằng 
sinh thái hoặc ngăn chặn suy thoái đất.) 
Example: Afforestation helps rehabilitate 
degraded hillsides. (Việc trồng rừng góp phần 
phục hồi những vùng đồi núi đã bị thoái hóa.) 
(adj): 
Biodegradable /ˌbaɪəʊdɪˈɡreɪdəbl/ 
Capable of being broken down 
naturally by microorganisms into harmless 
substances, 
reducing 
environmental 
pollution. (Tính từ dùng để chỉ khả năng 
phân hủy tự nhiên của vật chất thông qua 
hoạt động của vi sinh vật, giúp giảm thiểu ô 
nhiễm môi trường.) 
Example: Biodegradable packaging eases 
the plastic waste burden. (Bao bì phân hủy 
sinh học giúp giảm gánh nặng từ rác thải 
nhựa.) 
Barren /ˈbærən/ (adj): Incapable of 
supporting 
vegetation 
or 
agricultural 
production due to poor soil quality, 
environmental damage, or excessive chemical 
use. (Tính từ dùng để chỉ tình trạng đất đai 
không thể trồng trọt hoặc nuôi dưỡng cây cối 
do chất đất kém, môi trường bị phá hủy hoặc 
do lạm dụng hóa chất.) 
Example: Excessive pesticides left the 
farmland virtually barren. (Việc lạm dụng 
thuốc trừ sâu đã khiến cánh đồng gần như 
cằn cỗi.) 
Compost /ˈkɒmpɒst/ (verb): To 
convert organic matter such as food scraps or 
plant waste into nutrient-rich fertilizer 
through controlled decomposition. (Động từ 
dùng để chỉ hành động xử lý chất thải hữu 
cơ, như thức ăn thừa hoặc lá cây, bằng cách 
ủ để phân hủy thành phân bón giàu dinh 
dưỡng.) 
Example: Households are encouraged to 
compost organic waste. (Các hộ gia đình 
được khuyến khích ủ phân từ rác hữu cơ.)"""

    # Chia nho van ban
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=512,
        chunk_overlap=50,
        length_function=len

    )

    chunks = text_splitter.split_text(raw_text)

    # Embeding
    embedding_model = GPT4AllEmbeddings(model_file = "models/all-MiniLM-L6-v2-f16.gguf")

    # Dua vao Faiss Vector DB
    db = FAISS.from_texts(texts=chunks, embedding=embedding_model)
    db.save_local(vector_db_path)
    return db


def create_db_from_files():
    # Khai bao loader de quet toan bo thu muc dataa
    loader = DirectoryLoader(pdf_data_path, glob="*.pdf", loader_cls = PyPDFLoader)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    # Embeding
    embedding_model = GPT4AllEmbeddings(model_file="models/all-MiniLM-L6-v2-f16.gguf")
    db = FAISS.from_documents(chunks, embedding_model)
    db.save_local(vector_db_path)
    return db


create_db_from_files()