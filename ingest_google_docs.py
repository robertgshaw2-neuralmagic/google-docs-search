import os.path, argparse
from apis.embedding_model import EmbeddingModel
from apis.embedding_db import EmbeddingDB

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

parser = argparse.ArgumentParser(
                    prog='ingest_google_docs.py',
                    description='Program to download google docs, embed with OpenAI, and save in ChromaDB',
                    epilog='Contact robertgshaw2@gmail.com for help')
parser.add_argument('--maximum_items', type=int, default=1000, help='maximum number of documents to save in the database')
parser.add_argument('--reset_database', action='store_true', help='flag to reset the database before running the embedding process')

# authentication to google docs / google drive aPI
def authenticate():
    SCOPES = ['https://www.googleapis.com/auth/drive.metadata.readonly',
              'https://www.googleapis.com/auth/documents.readonly']
    creds = None

    # look to see if there are creds available
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    return creds

# parses content from google docs API
def read_structural_elements(elements):
    def read_paragraph_element(element):
        text_run = element.get('textRun')
        if not text_run:
            return ''
        return text_run.get('content')

    text = ''
    for value in elements:
        if 'paragraph' in value:
            elements = value.get('paragraph').get('elements', [])
            for elem in elements:
                text += read_paragraph_element(elem)
        elif 'table' in value:
            # The text in table cells are in nested Structural Elements and tables may be nested.
            table = value.get('table')
            for row in table.get('tableRows', []):
                cells = row.get('tableCells', [])
                for cell in cells:
                    text += read_structural_elements(cell.get('content'))
        elif 'tableOfContents' in value:
            # The text in the TOC is also in a Structural Element.
            toc = value.get('tableOfContents')
            text += read_structural_elements(toc.get('content'))
    return text

# main
def main(reset_database=False, maximum_items=1000):
    # authenticate and setup services
    print("Authenticating to Google Drive...")
    creds = authenticate()
    drive_service = build('drive', 'v3', credentials=creds)
    docs_service = build('docs', 'v1', credentials=creds)

    # setup embedding database and model
    print("Setting up model...")
    embedding_model = EmbeddingModel()
    print("Setting up database...")
    embedding_db    = EmbeddingDB()

    # embed until we hit the maxium
    print("Embedding documents...")
    item_count = 0
    next_page_token = None
    while item_count < maximum_items:
        
        # get next 50 google documents
        response = drive_service.files().list(
            q="mimeType='application/vnd.google-apps.document'",
            pageSize=50,
            pageToken = next_page_token,
            fields="nextPageToken, files(id, name)").execute()
        files = response.get('files', [])
        next_page_token = response.get('nextPageToken', None)

        # extract content from the files
        documents = []
        doc_metadatas = []
        for file in files:
            document = docs_service.documents().get(documentId=file['id']).execute() 
            documents.append(read_structural_elements(document.get('body').get('content')))
            doc_metadatas.append({
                'gdoc_id': file['id'],
                'gdoc_title': document['title'],
            })

        # embed documents and insert into database
        document_chunks, embeddings, indexes, chunk_ids = embedding_model.embed_documents(documents)
        embedding_db.insert_documents(
            embeddings=embeddings,
            document_chunks=document_chunks,
            doc_metadatas=doc_metadatas,
            indexes=indexes,
            chunk_ids=chunk_ids,
        )
        
        if next_page_token is None:
            break
        item_count += len(files)

        print(f"ITEMS_SAVED // MAXIMUM_ITEMS : {item_count} // {maximum_items}")

if __name__ == '__main__':
    args = parser.parse_args()
    print(f"Embedding up to {args.maximum_items} documents from Google Drive")
    main(
        reset_database=args.reset_database, 
        maximum_items=args.maximum_items,)
