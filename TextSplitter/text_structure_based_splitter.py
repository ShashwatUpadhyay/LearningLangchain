from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=0)

text = '''
Tool Introduction: Text Splitter Visualizer
The Text Splitter Visualizer is an innovative tool designed to help users understand and visualize the process of text splitting. Whether you are a writer, student, or professional, this tool enhances your ability to manage and analyze text efficiently.
Key Features:


1. User-Friendly Interface: The intuitive design allows users to easily input text and select various splitting methods, making it accessible for everyone.
2. Multiple Splitter Options: Choose from a variety of text splitting techniques, including character-based, recursive, token-based, and more. Each method is tailored to meet different text processing needs.
3. Real-Time Visualization: As you input text and adjust parameters, the tool provides immediate visual feedback, allowing you to see how your text is divided into chunks.
4. Customizable Parameters: Users can adjust settings such as chunk size, overlap, and separators to fine-tune the splitting process according to their specific requirements.
5. Performance Metrics: The tool displays important metrics, including total characters, number of chunks, and average chunk size, helping users gauge the effectiveness of their text splitting.
6. Language Support: The Text Splitter Visualizer supports multiple languages, making it a versatile tool for a global audience.
7. Educational Resource: Ideal for learning and teaching purposes, the tool provides explanations of different text splitting methods, enhancing users' understanding of text processing concepts.
The Text Splitter Visualizer is not just a tool; it's a powerful companion for anyone looking to improve their text management skills. Embrace the power of efficient text processing and transform the way you interact with text today!
'''

chunks = splitter.split_text(text)

print(len(chunks))
print(chunks)