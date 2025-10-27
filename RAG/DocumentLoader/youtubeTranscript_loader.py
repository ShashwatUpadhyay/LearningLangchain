from langchain_community.document_loaders import YoutubeLoader

loader = YoutubeLoader.from_youtube_url(
    youtube_url='https://www.youtube.com/watch?v=t8aSqlC_Duo',
    add_video_info=False
)

docs = loader.load()

print(len(docs))
print(docs)