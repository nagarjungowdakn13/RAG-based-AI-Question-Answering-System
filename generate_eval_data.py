import json
from pathlib import Path

def generate_data():
    eval_dir = Path("data/eval")
    docs_dir = Path("data/docs")
    hotpotqa_docs_dir = docs_dir / "hotpotqa"
    nq_docs_dir = docs_dir / "nq"

    eval_dir.mkdir(parents=True, exist_ok=True)
    hotpotqa_docs_dir.mkdir(parents=True, exist_ok=True)
    nq_docs_dir.mkdir(parents=True, exist_ok=True)

    # ─── HOTPOTQA (30 Multi-Hop Questions) ──────────────────────────────────
    # Format: {"question": ..., "answer": ..., "sources": [filename1, filename2]}
    hotpotqa_qa = [
        {
            "question": "Was Albert Einstein's first wife born before or after him?",
            "answer": "Mileva Marić was born before Albert Einstein (1875 vs 1879).",
            "sources": ["doc_einstein.txt", "doc_maric.txt"]
        },
        {
            "question": "Which AI lab did the co-creator of the Transformer, Ashish Vaswani, co-found in 2022?",
            "answer": "Adept AI Labs",
            "sources": ["doc_transformer.txt", "doc_vaswani.txt"]
        },
        {
            "question": "On what river is the capital of the country where Google DeepMind was founded situated?",
            "answer": "River Thames",
            "sources": ["doc_deepmind.txt", "doc_london.txt"]
        },
        {
            "question": "Which film directed by Christopher Nolan starred the actor who played J. Robert Oppenheimer?",
            "answer": "Oppenheimer starred Cillian Murphy, who also starred in Inception.",
            "sources": ["doc_oppenheimer.txt", "doc_nolan.txt"]
        },
        {
            "question": "Is the author of 'The Hobbit' older or younger than the creator of Narnia?",
            "answer": "J.R.R. Tolkien (Hobbit, b. 1892) is older than C.S. Lewis (Narnia, b. 1898).",
            "sources": ["doc_hobbit.txt", "doc_narnia.txt"]
        },
        {
            "question": "What is the capital of the state where the tech company that created the iPhone is headquartered?",
            "answer": "Sacramento (California, where Apple is headquartered).",
            "sources": ["doc_iphone.txt", "doc_california.txt"]
        },
        {
            "question": "Was the developer of the Python programming language born in the same country as the creator of Linux?",
            "answer": "No, Guido van Rossum was born in the Netherlands, whereas Linus Torvalds was born in Finland.",
            "sources": ["doc_python.txt", "doc_linux.txt"]
        },
        {
            "question": "Which university was attended by both the founders of Google?",
            "answer": "Stanford University",
            "sources": ["doc_google.txt", "doc_stanford.txt"]
        },
        {
            "question": "What is the native language of the country where the designer of the Eiffel Tower was born?",
            "answer": "French (Gustave Eiffel was born in France).",
            "sources": ["doc_eiffel.txt", "doc_france.txt"]
        },
        {
            "question": "Does the company that developed the ChatGPT model have its main headquarters in the capital of the USA?",
            "answer": "No, OpenAI is based in San Francisco, California, not Washington D.C.",
            "sources": ["doc_openai.txt", "doc_usa.txt"]
        },
        {
            "question": "Was the author of the theory of evolution born in the same century as the author of 'Principia Mathematica'?",
            "answer": "No, Charles Darwin was born in the 19th century (1809), while Isaac Newton published Principia in the 17th century (1687).",
            "sources": ["doc_evolution.txt", "doc_principia.txt"]
        },
        {
            "question": "What is the capital of the country where the inventor of the printing press was born?",
            "answer": "Berlin (Johannes Gutenberg was born in Germany).",
            "sources": ["doc_printing.txt", "doc_germany.txt"]
        },
        {
            "question": "Which country is home to the mountain range where the highest peak on Earth is located?",
            "answer": "Nepal (Home of Mount Everest in the Himalayas).",
            "sources": ["doc_everest.txt", "doc_nepal.txt"]
        },
        {
            "question": "What is the official currency of the country where the company that makes the PlayStation is headquartered?",
            "answer": "Japanese Yen (Sony is headquartered in Japan).",
            "sources": ["doc_playstation.txt", "doc_japan.txt"]
        },
        {
            "question": "Was the composer of the 'Moonlight Sonata' born in the same city as the composer of the 'Messiah'?",
            "answer": "No, Beethoven (Moonlight Sonata) was born in Bonn, whereas Handel (Messiah) was born in Halle.",
            "sources": ["doc_beethoven.txt", "doc_handel.txt"]
        },
        {
            "question": "Which space agency launched the telescope that succeeded the Hubble Space Telescope?",
            "answer": "NASA (along with ESA and CSA) launched the James Webb Space Telescope.",
            "sources": ["doc_hubble.txt", "doc_jwst.txt"]
        },
        {
            "question": "What ocean is adjacent to the state where Microsoft was founded?",
            "answer": "Pacific Ocean (Washington State)",
            "sources": ["doc_microsoft.txt", "doc_washington.txt"]
        },
        {
            "question": "Which country hosts the headquarters of the organization that awards the Nobel Prizes?",
            "answer": "Sweden",
            "sources": ["doc_nobel.txt", "doc_sweden.txt"]
        },
        {
            "question": "Was the designer of the first compiler born in the same country as the designer of the Analytical Engine?",
            "answer": "No, Grace Hopper (first compiler) was born in the USA, while Charles Babbage (Analytical Engine) was born in England.",
            "sources": ["doc_compiler.txt", "doc_analytical.txt"]
        },
        {
            "question": "What is the currency of the nation where the Suez Canal is located?",
            "answer": "Egyptian Pound",
            "sources": ["doc_suez.txt", "doc_egypt.txt"]
        },
        {
            "question": "Was the artist who painted 'The Starry Night' from the same country as the painter of 'Guernica'?",
            "answer": "No, Vincent van Gogh (Starry Night) was Dutch, whereas Pablo Picasso (Guernica) was Spanish.",
            "sources": ["doc_starry.txt", "doc_guernica.txt"]
        },
        {
            "question": "What is the capital of the country where the company that produces Spotify is based?",
            "answer": "Stockholm (Sweden)",
            "sources": ["doc_spotify.txt", "doc_sweden.txt"]
        },
        {
            "question": "Did the actor who played Iron Man in the MCU star in the 2023 film 'Oppenheimer'?",
            "answer": "Yes, Robert Downey Jr. played Lewis Strauss in Oppenheimer.",
            "sources": ["doc_ironman.txt", "doc_oppenheimer.txt"]
        },
        {
            "question": "Which ocean border is closest to the headquarters of the European Space Agency?",
            "answer": "Atlantic Ocean (ESA is headquartered in Paris, France).",
            "sources": ["doc_esa.txt", "doc_france.txt"]
        },
        {
            "question": "Is the capital of the country that hosted the 2008 Summer Olympics located on the coast?",
            "answer": "No, Beijing (China) is not a coastal city.",
            "sources": ["doc_olympics.txt", "doc_china.txt"]
        },
        {
            "question": "Which programming language was developed first: the one created by Bjarne Stroustrup or the one by Sun Microsystems?",
            "answer": "C++ (Bjarne Stroustrup, 1985) was created before Java (Sun Microsystems, 1995).",
            "sources": ["doc_cpp.txt", "doc_java.txt"]
        },
        {
            "question": "What is the capital city of the country that spans two continents and contains the Bosporus Strait?",
            "answer": "Ankara (Turkey)",
            "sources": ["doc_strait.txt", "doc_turkey.txt"]
        },
        {
            "question": "Was the discoverer of penicillin born in the same country as the discoverer of radium?",
            "answer": "No, Alexander Fleming (penicillin) was born in Scotland, while Marie Curie (radium) was born in Poland.",
            "sources": ["doc_penicillin.txt", "doc_radium.txt"]
        },
        {
            "question": "What is the name of the river that flows through the capital of Egypt?",
            "answer": "Nile River (flows through Cairo)",
            "sources": ["doc_cairo.txt", "doc_nile.txt"]
        },
        {
            "question": "Was the architect of the Guggenheim Museum in Bilbao born in the USA?",
            "answer": "Yes, Frank Gehry was born in Canada but is naturalized US and based in USA, or Frank Lloyd Wright designed the NY one. Frank Gehry designed the Bilbao museum.",
            "sources": ["doc_guggenheim.txt", "doc_gehry.txt"]
        }
    ]

    # Write HotpotQA documents
    hotpotqa_docs = {
        "doc_einstein.txt": "Albert Einstein (14 March 1879 – 18 April 1955) was a German-born theoretical physicist. He is widely acknowledged as one of the greatest physicists of all time. He married Mileva Marić in 1903.",
        "doc_maric.txt": "Mileva Marić (19 December 1875 – 4 August 1948) was a Serbian physicist and mathematician. She was the first wife of Albert Einstein and co-collaborator on early scientific works.",
        "doc_transformer.txt": "The Transformer architecture was introduced in the landmark paper 'Attention Is All You Need' in 2017 by Google researchers, including Ashish Vaswani, Jakob Uszkoreit, and others.",
        "doc_vaswani.txt": "Ashish Vaswani is a prominent AI researcher who worked at Google Brain. In 2022, he co-founded Adept AI Labs, a startup focused on building general intelligence agents.",
        "doc_deepmind.txt": "Google DeepMind is a world-leading artificial intelligence research laboratory. It was founded in London, England in 2010 by Demis Hassabis, Shane Legg, and Mustafa Suleyman.",
        "doc_london.txt": "London is the capital and largest city of England and the United Kingdom. It is situated on the River Thames in the south-east of Great Britain.",
        "doc_oppenheimer.txt": "J. Robert Oppenheimer was an American theoretical physicist and director of the Manhattan Project's Los Alamos Laboratory. He was portrayed by Cillian Murphy in the 2023 film Oppenheimer.",
        "doc_nolan.txt": "Christopher Nolan is a British-American filmmaker known for Hollywood blockbusters. He directed the mind-bending film Inception (2010) and Oppenheimer (2023), both featuring Cillian Murphy.",
        "doc_hobbit.txt": "The Hobbit is a children's fantasy novel written by English author J.R.R. Tolkien. Tolkien was born in 1892 and worked as a professor at the University of Oxford.",
        "doc_narnia.txt": "The Chronicles of Narnia is a series of seven fantasy novels by C.S. Lewis. Lewis was born in 1898 and was a close friend of J.R.R. Tolkien.",
        "doc_iphone.txt": "The iPhone is a line of smartphones designed and marketed by Apple Inc. Apple is a global technology company headquartered in Cupertino, California.",
        "doc_california.txt": "California is a state in the Western United States. Its capital is Sacramento, while its largest city is Los Angeles, and the tech hub of Silicon Valley is in the north.",
        "doc_python.txt": "Python is a high-level programming language created by Guido van Rossum. Van Rossum was born in the Netherlands in 1956 and released Python in 1991.",
        "doc_linux.txt": "Linux is a family of open-source Unix-like operating systems based on the Linux kernel, created by Linus Torvalds. Torvalds was born in Finland in 1969.",
        "doc_google.txt": "Google was founded in September 1998 by Larry Page and Sergey Brin while they were Ph.D. students at a California university. They incorporated it as a privately held company.",
        "doc_stanford.txt": "Stanford University is a prestigious private research university in Stanford, California. It was attended by Larry Page and Sergey Brin, who created Google there.",
        "doc_eiffel.txt": "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris. It is named after the engineer Gustave Eiffel, whose company designed and built the tower.",
        "doc_france.txt": "France is a country located in Western Europe. Its capital is Paris, its language is French, and it is known for landmarks like the Eiffel Tower and the Louvre.",
        "doc_openai.txt": "OpenAI is an AI research organization. Founded in 2015 and headquartered in San Francisco, California, it developed the widely popular ChatGPT conversational model.",
        "doc_usa.txt": "The United States of America (USA) is a country primarily located in North America. Its federal capital is Washington, D.C., and its largest city is New York City.",
        "doc_evolution.txt": "The theory of evolution by natural selection was formulated by Charles Darwin. Darwin was an English naturalist born in 1809 and published 'On the Origin of Species' in 1859.",
        "doc_principia.txt": "The Philosophiæ Naturalis Principia Mathematica is a work in three books by Sir Isaac Newton, first published in 1687. It established the laws of classical mechanics.",
        "doc_printing.txt": "The printing press was invented by Johannes Gutenberg around 1440. Gutenberg was a German blacksmith and publisher who introduced printing to Europe.",
        "doc_germany.txt": "Germany is a country in Central Europe. Its capital is Berlin, and it is known for historical figures like Gutenberg, Einstein, and Beethoven.",
        "doc_everest.txt": "Mount Everest is Earth's highest mountain above sea level, located in the Mahalangur Himal sub-range of the Himalayas. The international border runs across its summit.",
        "doc_nepal.txt": "Nepal is a landlocked country in South Asia, located mainly in the Himalayas. It shares borders with China and India and is home to Mount Everest.",
        "doc_playstation.txt": "The PlayStation is a series of video game consoles created and developed by Sony Interactive Entertainment, which is headquartered in Tokyo, Japan.",
        "doc_japan.txt": "Japan is an island country in East Asia. Its capital is Tokyo, and its official currency is the Japanese Yen. It is a leader in technology and gaming.",
        "doc_beethoven.txt": "Ludwig van Beethoven was a German composer and pianist. He composed the Moonlight Sonata in 1801. He was born in the German city of Bonn.",
        "doc_handel.txt": "George Frideric Handel was a German-British Baroque composer famous for operas and oratorios, especially 'Messiah' (1741). He was born in Halle, Germany.",
        "doc_hubble.txt": "The Hubble Space Telescope is a space telescope that was launched into low Earth orbit in 1990. It was built by the United States space agency NASA.",
        "doc_jwst.txt": "The James Webb Space Telescope (JWST) was launched in 2021 as the successor to Hubble. It was developed by NASA in collaboration with ESA and CSA.",
        "doc_microsoft.txt": "Microsoft Corporation is an American multinational technology corporation. It was founded by Bill Gates and Paul Allen in 1975, originally in Albuquerque but later moved to Washington State.",
        "doc_washington.txt": "Washington is a state in the Pacific Northwest region of the Western United States. It borders the Pacific Ocean and houses tech giants like Microsoft and Amazon.",
        "doc_nobel.txt": "The Nobel Prizes are five separate prizes awarded according to Alfred Nobel's will. The Nobel Foundation is located in Stockholm, where most prizes are presented.",
        "doc_sweden.txt": "Sweden is a Scandinavian nation in Northern Europe. Its capital is Stockholm, and it is the home of Spotify and the Nobel Prize organization.",
        "doc_compiler.txt": "The first compiler was created by Grace Hopper in 1952 for the A-0 System programming language. Hopper was a US Navy rear admiral and computer scientist.",
        "doc_analytical.txt": "The Analytical Engine was a proposed mechanical general-purpose computer designed by English mathematician Charles Babbage. It was first described in 1837.",
        "doc_suez.txt": "The Suez Canal is an artificial sea-level waterway in Egypt, connecting the Mediterranean Sea to the Red Sea. It was opened in 1869.",
        "doc_egypt.txt": "Egypt is a transcontinental country spanning the northeast corner of Africa and southwest corner of Asia. Its currency is the Egyptian Pound.",
        "doc_starry.txt": "The Starry Night is an oil-on-canvas painting by the Dutch Post-Impressionist painter Vincent van Gogh. Painted in June 1889, it depicts the view from his asylum room.",
        "doc_guernica.txt": "Guernica is a large 1937 oil painting by Spanish artist Pablo Picasso. It is one of his best-known works, depicting the tragedies of war.",
        "doc_spotify.txt": "Spotify is a proprietary Swedish audio streaming and media services provider. It was founded in 2006 by Daniel Ek and Martin Lorentzon.",
        "doc_ironman.txt": "Iron Man (Tony Stark) is a fictional superhero in the Marvel Cinematic Universe, portrayed by Robert Downey Jr. from 2008 until 2019.",
        "doc_esa.txt": "The European Space Agency (ESA) is an intergovernmental organization dedicated to the exploration of space. It is headquartered in Paris, France.",
        "doc_olympics.txt": "The 2008 Summer Olympics were held in Beijing, China. It was a massive multi-sport event featuring athletes from all over the world.",
        "doc_china.txt": "China is a country in East Asia. Its capital is Beijing, its largest city is Shanghai, and it is the most populous country in the world.",
        "doc_cpp.txt": "C++ is a general-purpose programming language created by Bjarne Stroustrup as an extension of the C programming language. It was released in 1985.",
        "doc_java.txt": "Java is a class-based, object-oriented programming language developed by James Gosling at Sun Microsystems and released in 1995.",
        "doc_strait.txt": "The Bosporus Strait is a narrow, natural strait and an internationally significant waterway located in northwestern Turkey. It connects the Black Sea with the Sea of Marmara.",
        "doc_turkey.txt": "Turkey is a transcontinental country located mainly on the Anatolian Peninsula in Western Asia. Its capital is Ankara, and its largest city is Istanbul.",
        "doc_penicillin.txt": "Penicillin was discovered by Scottish physician and microbiologist Alexander Fleming in 1928 at St. Mary's Hospital, London.",
        "doc_radium.txt": "Radium was discovered by Polish physicist Marie Curie and her husband Pierre Curie in 1898. Marie Curie was born in Warsaw, Poland.",
        "doc_cairo.txt": "Cairo is the capital of Egypt and the largest metropolitan area in the Arab world. It is situated on the Nile River.",
        "doc_nile.txt": "The Nile is a major north-flowing river in northeastern Africa. It is the longest river in Africa and historically considered the longest river in the world.",
        "doc_guggenheim.txt": "The Guggenheim Museum Bilbao is a museum of modern and contemporary art designed by architect Frank Gehry, located in Bilbao, Spain.",
        "doc_gehry.txt": "Frank Gehry is a Canadian-born American architect. A number of his buildings, including his private residence in Santa Monica, California, have become world-renowned attractions."
    }

    # Add 10 distractor documents for HotpotQA to make retrieval realistic
    for i in range(10):
        name = f"distract_hotpot_{i}.txt"
        hotpotqa_docs[name] = f"This is a distractor document {i} for the HotpotQA evaluation corpus. It talks about generic topics such as weather forecast, cooking recipes, and gardening tips in spring."

    # Write HotpotQA doc files
    for filename, content in hotpotqa_docs.items():
        (hotpotqa_docs_dir / filename).write_text(content, encoding="utf-8")

    with open(eval_dir / "hotpotqa_subset.json", "w", encoding="utf-8") as f:
        json.dump(hotpotqa_qa, f, indent=2, ensure_ascii=False)


    # ─── NATURAL QUESTIONS (20 Single-Hop Questions) ────────────────────────
    nq_qa = [
        {
            "question": "Who painted the Mona Lisa?",
            "answer": "Leonardo da Vinci",
            "sources": ["doc_mona_lisa.txt"]
        },
        {
            "question": "What is the height of Mount Everest in meters?",
            "answer": "8,848 meters",
            "sources": ["doc_mount_everest_nq.txt"]
        },
        {
            "question": "Who developed the theory of general relativity?",
            "answer": "Albert Einstein",
            "sources": ["doc_relativity.txt"]
        },
        {
            "question": "What is the capital city of Australia?",
            "answer": "Canberra",
            "sources": ["doc_canberra.txt"]
        },
        {
            "question": "In what year did the Titanic sink?",
            "answer": "1912",
            "sources": ["doc_titanic.txt"]
        },
        {
            "question": "Who wrote the play Romeo and Juliet?",
            "answer": "William Shakespeare",
            "sources": ["doc_shakespeare.txt"]
        },
        {
            "question": "What is the chemical symbol for gold?",
            "answer": "Au",
            "sources": ["doc_gold_element.txt"]
        },
        {
            "question": "Which planet is known as the Red Planet?",
            "answer": "Mars",
            "sources": ["doc_mars_planet.txt"]
        },
        {
            "question": "Who was the first president of the United States?",
            "answer": "George Washington",
            "sources": ["doc_washington_president.txt"]
        },
        {
            "question": "What is the longest river in the world?",
            "answer": "Nile River",
            "sources": ["doc_nile_river.txt"]
        },
        {
            "question": "What is the speed of light in vacuum?",
            "answer": "299,792,458 meters per second",
            "sources": ["doc_speed_light.txt"]
        },
        {
            "question": "Who is the author of the Harry Potter books?",
            "answer": "J.K. Rowling",
            "sources": ["doc_rowling.txt"]
        },
        {
            "question": "Which element has the atomic number 1?",
            "answer": "Hydrogen",
            "sources": ["doc_hydrogen.txt"]
        },
        {
            "question": "What is the capital of Spain?",
            "answer": "Madrid",
            "sources": ["doc_madrid.txt"]
        },
        {
            "question": "In which ocean is the Mariana Trench located?",
            "answer": "Pacific Ocean",
            "sources": ["doc_mariana.txt"]
        },
        {
            "question": "Who discovered gravity when an apple fell on his head?",
            "answer": "Sir Isaac Newton",
            "sources": ["doc_newton_apple.txt"]
        },
        {
            "question": "What gas do plants absorb during photosynthesis?",
            "answer": "Carbon dioxide",
            "sources": ["doc_photosynthesis.txt"]
        },
        {
            "question": "Which country is the largest by land area?",
            "answer": "Russia",
            "sources": ["doc_russia_country.txt"]
        },
        {
            "question": "In which city are the headquarters of the United Nations located?",
            "answer": "New York City",
            "sources": ["doc_un_hq.txt"]
        },
        {
            "question": "Who is the founder of Microsoft?",
            "answer": "Bill Gates",
            "sources": ["doc_gates_founder.txt"]
        }
    ]

    # Write NQ documents
    nq_docs = {
        "doc_mona_lisa.txt": "The Mona Lisa is a half-length portrait painting by Italian artist Leonardo da Vinci. It is considered an archetypal masterpiece of the Italian Renaissance.",
        "doc_mount_everest_nq.txt": "Mount Everest is Earth's highest mountain above sea level, located in the Himalayas. The summit is official at 8,848 meters high.",
        "doc_relativity.txt": "General relativity is the geometric theory of gravitation published by Albert Einstein in 1915 and the current description of gravitation in modern physics.",
        "doc_canberra.txt": "Canberra is the capital city of Australia. It is Australia's largest inland city and the eighth-largest city overall, located in the Australian Capital Territory.",
        "doc_titanic.txt": "RMS Titanic was a British passenger liner operated by the White Star Line that sank in the North Atlantic Ocean on 15 April 1912 after striking an iceberg.",
        "doc_shakespeare.txt": "Romeo and Juliet is a tragedy written by William Shakespeare early in his career about two young Italian star-crossed lovers whose deaths ultimately reconcile their feuding families.",
        "doc_gold_element.txt": "Gold is a chemical element with the symbol Au (from Latin: aurum) and atomic number 79. In its pure form, it is a bright, slightly reddish yellow, dense, soft, malleable, and ductile metal.",
        "doc_mars_planet.txt": "Mars is the fourth planet from the Sun and the second-smallest planet in the Solar System. It is often referred to as the 'Red Planet' due to iron oxide on its surface.",
        "doc_washington_president.txt": "George Washington was an American military officer, statesman, and Founding Father who served as the first president of the United States from 1789 to 1797.",
        "doc_nile_river.txt": "The Nile is a major north-flowing river in northeastern Africa. It is the longest river in Africa and historically considered the longest river in the world.",
        "doc_speed_light.txt": "The speed of light in vacuum, commonly denoted c, is a universal physical constant. Its exact value is defined as 299,792,458 meters per second.",
        "doc_rowling.txt": "Joanne Rowling, better known by her pen name J.K. Rowling, is a British author who wrote Harry Potter, a seven-volume fantasy novel series.",
        "doc_hydrogen.txt": "Hydrogen is the chemical element with the symbol H and atomic number 1. It is the lightest element and the most abundant chemical substance in the Universe.",
        "doc_madrid.txt": "Madrid is the capital and most populous city of Spain. The city has almost 3.4 million inhabitants and is located on the Manzanares River.",
        "doc_mariana.txt": "The Mariana Trench is located in the western Pacific Ocean. It is the deepest oceanic trench on Earth, reaching a maximum depth of nearly 11,000 meters.",
        "doc_newton_apple.txt": "Sir Isaac Newton was an English mathematician and physicist. According to legend, he formulated his theory of universal gravitation after watching an apple fall from a tree.",
        "doc_photosynthesis.txt": "Photosynthesis is a process used by plants and other organisms to convert light energy into chemical energy. During this process, plants absorb carbon dioxide and release oxygen.",
        "doc_russia_country.txt": "Russia is a transcontinental country spanning Eastern Europe and Northern Asia. It is the largest country in the world by area, covering over 17 million square kilometers.",
        "doc_un_hq.txt": "The headquarters of the United Nations is a complex in New York City, New York, designed by a board of architects led by Wallace Harrison.",
        "doc_gates_founder.txt": "Bill Gates is an American businessman and philanthropist. He co-founded Microsoft Corporation with his childhood friend Paul Allen in 1975."
    }

    # Add 10 distractor documents for NQ
    for i in range(10):
        name = f"distract_nq_{i}.txt"
        nq_docs[name] = f"This is a distractor document {i} for the Natural Questions evaluation corpus. It talks about topics such as space exploration, deep sea biology, and classic literature."

    # Write NQ doc files
    for filename, content in nq_docs.items():
        (nq_docs_dir / filename).write_text(content, encoding="utf-8")

    with open(eval_dir / "nq_subset.json", "w", encoding="utf-8") as f:
        json.dump(nq_qa, f, indent=2, ensure_ascii=False)

    print("Curated HotpotQA and NQ-open datasets generated successfully!")

if __name__ == "__main__":
    generate_data()
