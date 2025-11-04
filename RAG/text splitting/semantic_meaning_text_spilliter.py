from langchain_text_splitters import   RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


loader= PyPDFLoader("CSL7590_C18-1.pdf")

docs= loader.load()

text="""A black hole is an astronomical body so compact that its gravity prevents anything from escaping, even light. Albert Einstein's theory of general relativity predicts that a sufficiently compact mass will form a black hole.[4] The boundary of no escape is called the event horizon. In general relativity, a black hole's event horizon seals an object's fate but produces no locally detectable change when crossed.[5] In many ways, a black hole acts like an ideal black body, as it reflects no light.[6][7] Quantum field theory in curved spacetime predicts that event horizons emit Hawking radiation, with the same spectrum as a black body of a temperature inversely proportional to its mass. This temperature is of the order of billionths of a kelvin for stellar black holes, making it essentially impossible to observe directly.

Objects whose gravitational fields are too strong for light to escape were first considered in the 18th century by John Michell and Pierre-Simon Laplace. In 1916, Karl Schwarzschild found the first modern solution of general relativity that would characterise a black hole. Due to his influential research, the Schwarzschild metric is named after him. David Finkelstein, in 1958, first published the interpretation of "black hole" as a region of space from which nothing can escape. Black holes were long considered a mathematical curiosity; it was not until the 1960s that theoretical work showed they were a generic prediction of general relativity. The first black hole known was Cygnus X-1, identified by several researchers independently in 1971.[8][9]

Black holes typically form when massive stars collapse at the end of their life cycle. After a black hole has formed, it can grow by absorbing mass from its surroundings. Supermassive black holes of millions of solar masses may form by absorbing other stars and merging with other black holes, or via direct collapse of gas clouds. There is consensus that supermassive black holes exist in the centres of most galaxies.

The presence of a black hole can be inferred through its interaction with other matter and with electromagnetic radiation such as visible light. Matter falling toward a black hole can form an accretion disk of infalling plasma, heated by friction and emitting light. In extreme cases, this creates a quasar, some of the brightest objects in the universe. Stars passing too close to a supermassive black hole can be shredded into streamers that shine very brightly before being "swallowed".[10] If other stars are orbiting a black hole, their orbits can be used to determine the black hole's mass and location. Such observations can be used to exclude possible alternatives such as neutron stars. In this way, astronomers have identified numerous stellar black hole candidates in binary systems and established that the radio source known as Sagittarius A*, at the core of the Milky Way galaxy, contains a supermassive black hole of about 4.3 million solar masses

Weather refers to the state of the Earth's atmosphere at a specific place and time, typically described in terms of temperature, humidity, cloud cover, and stability.[1] On Earth, most weather phenomena occur in the lowest layer of the planet's atmosphere, the troposphere,[2][3] just below the stratosphere. Weather refers to day-to-day temperature, precipitation, and other atmospheric conditions, whereas climate is the term for the averaging of atmospheric conditions over longer periods of time.[4] When used without qualification, "weather" is generally understood to mean the weather of Earth.

Weather is driven by air pressure, temperature, and moisture differences between one place and another. These differences can occur due to the Sun's angle at any particular spot, which varies with latitude. The strong temperature contrast between polar and tropical air gives rise to the largest scale atmospheric circulations: the Hadley cell, the Ferrel cell, the polar cell, and the jet stream. Weather systems in the middle latitudes, such as extratropical cyclones, are caused by instabilities of the jet streamflow. Because Earth's axis is tilted relative to its orbital plane (called the ecliptic), sunlight is incident at different angles at different times of the year. On Earth's surface, temperatures usually range ±40 °C (−40 °F to 104 °F) annually. Over thousands of years, changes in Earth's orbit can affect the amount and distribution of solar energy received by Earth, thus influencing long-term climate and global climate change. """


splitter = RecursiveCharacterTextSplitter (
    chunk_size=100,
    chunk_overlap=100,
)

result = splitter.split_text(text)

print(result)