"""
Real-Life Movie Data Generator for Movie Ratings Data Pipeline
Generates two separate raw datasets based on 110+ real-world iconic movies
(with authentic release years, genres, directors, runtimes, budgets, box office, studios,
and true IMDb/RottenTomatoes-calibrated audience rating distributions), plus intentional
real-world data pipeline anomalies:
1. raw_movie_details.csv - Real-life movie metadata with duplicate title entries
   (messy casing/whitespace/different IDs) and missing values to demonstrate ETL cleaning.
2. raw_movie_ratings.csv - 7,500+ individual audience rating logs calibrated to each real movie's
   true IMDb rating & vote volume, with intentional duplicate user ratings, orphan IDs, and outliers.
"""
import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
# seed represent 42 = which means the answer to life, the universe, and everything in Douglas Adams' "The Hitchhiker's Guide to the Galaxy"
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")

# 110 Real-Life Iconic Movies: 
# movies with authentic release years, genres, directors, runtimes, budgets, box office, studios,
# (Title, Year, Genres, Director, Runtime_min, Budget_MUSD, BoxOffice_MUSD, Language, Rating_Cert, Studio, True_IMDb_Mean, Popularity_Weight, Synopsis)
REAL_MOVIES = [
    ("The Shawshank Redemption", 1994, "Drama|Crime", "Frank Darabont", 142, 25.0, 73.3, "English", "R", "Castle Rock Entertainment", 9.3, 3.8, "Over the course of several years, two convicts form a friendship, seeking consolation and, eventually, redemption through basic compassion."),
    ("The Godfather", 1972, "Crime|Drama", "Francis Ford Coppola", 175, 6.0, 250.3, "English", "R", "Paramount Pictures", 9.2, 3.5, "The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant youngest son."),
    ("The Dark Knight", 2008, "Action|Crime|Drama", "Christopher Nolan", 152, 185.0, 1006.0, "English", "PG-13", "Warner Bros.", 9.0, 4.0, "When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological and physical tests."),
    ("The Godfather Part II", 1974, "Crime|Drama", "Francis Ford Coppola", 202, 13.0, 93.0, "English", "R", "Paramount Pictures", 9.0, 2.7, "The early life and career of Vito Corleone in 1920s New York City is portrayed, while his son, Michael, expands and tightens his grip on the family crime syndicate."),
    ("12 Angry Men", 1957, "Crime|Drama", "Sidney Lumet", 96, 0.35, 2.0, "English", "PG", "United Artists", 9.0, 2.2, "The jury in a New York City murder trial is frustrated by a single member whose skeptical caution forces them to more carefully consider the evidence."),
    ("Schindler's List", 1993, "Biography|Drama|History", "Steven Spielberg", 195, 22.0, 322.2, "English", "R", "Universal Pictures", 9.0, 2.8, "In German-occupied Poland during World War II, industrialist Oskar Schindler gradually becomes concerned for his Jewish workforce after witnessing their persecution."),
    ("The Lord of the Rings: The Return of the King", 2003, "Action|Adventure|Fantasy", "Peter Jackson", 201, 94.0, 1147.9, "English", "PG-13", "New Line Cinema", 9.0, 3.6, "Gandalf and Aragorn lead the World of Men against Sauron's army to draw his gaze from Frodo and Sam as they approach Mount Doom with the One Ring."),
    ("Pulp Fiction", 1994, "Crime|Drama", "Quentin Tarantino", 154, 8.5, 213.9, "English", "R", "Miramax", 8.9, 3.5, "The lives of two mob hitmen, a boxer, a gangster and his wife, and a pair of diner bandits intertwine in four tales of violence and redemption."),
    ("The Lord of the Rings: The Fellowship of the Ring", 2001, "Action|Adventure|Fantasy", "Peter Jackson", 178, 93.0, 898.2, "English", "PG-13", "New Line Cinema", 8.9, 3.5, "A meek Hobbit from the Shire and eight companions set out on a journey to destroy the powerful One Ring and save Middle-earth from the Dark Lord Sauron."),
    ("Forrest Gump", 1994, "Drama|Romance", "Robert Zemeckis", 142, 55.0, 678.2, "English", "PG-13", "Paramount Pictures", 8.8, 3.6, "The history of the United States from the 1950s to the '70s unfolds from the perspective of an Alabama man with an IQ of 75, who yearns to be reunited with his childhood sweetheart."),
    ("Fight Club", 1999, "Drama|Thriller", "David Fincher", 139, 63.0, 101.2, "English", "R", "20th Century Fox", 8.8, 3.4, "An insomniac office worker and a devil-may-care soap maker form an underground fight club that evolves into much more."),
    ("Inception", 2010, "Action|Adventure|Sci-Fi", "Christopher Nolan", 148, 160.0, 839.0, "English", "PG-13", "Warner Bros.", 8.8, 3.9, "A thief who steals corporate secrets through the use of dream-sharing technology is given the inverse task of planting an idea into the mind of a C.E.O."),
    ("The Lord of the Rings: The Two Towers", 2002, "Action|Adventure|Fantasy", "Peter Jackson", 179, 94.0, 947.9, "English", "PG-13", "New Line Cinema", 8.8, 3.2, "While Frodo and Sam edge closer to Mordor with the help of the shifty Gollum, the divided fellowship makes a stand against Sauron's new ally, Saruman."),
    ("Star Wars: Episode V - The Empire Strikes Back", 1980, "Action|Adventure|Fantasy", "Irvin Kershner", 124, 18.0, 538.4, "English", "PG", "Lucasfilm", 8.7, 2.9, "After the Rebels are overpowered by the Empire, Luke Skywalker begins his Jedi training with Yoda, while his friends are pursued across the galaxy."),
    ("The Matrix", 1999, "Action|Sci-Fi", "Lana Wachowski", 136, 63.0, 467.2, "English", "R", "Warner Bros.", 8.7, 3.5, "When a beautiful stranger leads computer hacker Neo to a forbidding underworld, he discovers the shocking truth--the life he knows is the elaborate deception of an evil cyber-intelligence."),
    ("Goodfellas", 1990, "Biography|Crime|Drama", "Martin Scorsese", 145, 25.0, 47.1, "English", "R", "Warner Bros.", 8.7, 2.7, "The story of Henry Hill and his life in the mafia, covering his relationship with his wife Karen and his mob partners Jimmy Conway and Tommy DeVito."),
    ("Interstellar", 2014, "Adventure|Drama|Sci-Fi", "Christopher Nolan", 169, 165.0, 733.2, "English", "PG-13", "Paramount Pictures", 8.7, 3.8, "When Earth becomes uninhabitable in the future, a farmer and ex-NASA pilot, Joseph Cooper, is tasked to pilot a spacecraft, along with a team of researchers, to find a new planet for humans."),
    ("Se7en", 1995, "Crime|Drama|Mystery", "David Fincher", 127, 33.0, 327.3, "English", "R", "New Line Cinema", 8.6, 2.9, "Two detectives, a rookie and a veteran, hunt a serial killer who uses the seven deadly sins as his motives."),
    ("The Silence of the Lambs", 1991, "Crime|Drama|Thriller", "Jonathan Demme", 118, 19.0, 272.7, "English", "R", "Orion Pictures", 8.6, 2.8, "A young F.B.I. cadet must receive the help of an incarcerated and manipulative cannibal killer to help catch another serial killer, a madman who skins his victims."),
    ("Saving Private Ryan", 1998, "Drama|War", "Steven Spielberg", 169, 70.0, 482.3, "English", "R", "DreamWorks Pictures", 8.6, 2.9, "Following the Normandy Landings, a group of U.S. soldiers go behind enemy lines to retrieve a paratrooper whose brothers have been killed in action."),
    ("Spirited Away", 2001, "Animation|Adventure|Family", "Hayao Miyazaki", 125, 19.0, 395.8, "Japanese", "PG", "Studio Ghibli", 8.6, 2.7, "During her family's move to the suburbs, a sullen 10-year-old girl wanders into a world ruled by gods, witches and spirits, models where humans are changed into beasts."),
    ("City of God", 2002, "Crime|Drama", "Fernando Meirelles", 130, 3.3, 30.6, "Portuguese", "R", "O2 Filmes", 8.6, 2.1, "In the slums of Rio, two kids' paths diverge as one struggles to become a photographer and the other a kingpin."),
    ("Parasite", 2019, "Drama|Thriller", "Bong Joon-ho", 132, 11.4, 262.6, "Korean", "R", "CJ Entertainment", 8.5, 3.2, "Greed and class discrimination threaten the newly formed symbiotic relationship between the wealthy Park family and the destitute Kim clan."),
    ("The Departed", 2006, "Crime|Drama|Thriller", "Martin Scorsese", 151, 90.0, 291.5, "English", "R", "Warner Bros.", 8.5, 2.8, "An undercover cop and a mole in the police attempt to identify each other while infiltrating an Irish gang in South Boston."),
    ("Whiplash", 2014, "Drama|Music", "Damien Chazelle", 106, 3.3, 49.0, "English", "R", "Sony Pictures Classics", 8.5, 2.7, "A promising young drummer enrolls at a cut-throat music conservatory where his dreams of greatness are mentored by an instructor who will stop at nothing."),
    ("Gladiator", 2000, "Action|Adventure|Drama", "Ridley Scott", 155, 103.0, 465.4, "English", "R", "DreamWorks Pictures", 8.5, 3.1, "A former Roman General sets out to exact vengeance against the corrupt emperor who murdered his family and sent him into slavery."),
    ("The Prestige", 2006, "Drama|Mystery|Sci-Fi", "Christopher Nolan", 130, 40.0, 109.7, "English", "PG-13", "Warner Bros.", 8.5, 2.9, "After a tragic accident, two stage magicians in 1890s London engage in a battle to create the ultimate illusion while sacrificing everything they have to outwit each other."),
    ("The Lion King", 1994, "Animation|Adventure|Drama", "Roger Allers", 88, 45.0, 968.5, "English", "G", "Walt Disney Pictures", 8.5, 2.8, "Lion prince Simba and his father are targeted by his bitter uncle, who wants to ascend the throne himself."),
    ("Dune: Part Two", 2024, "Action|Adventure|Sci-Fi", "Denis Villeneuve", 166, 190.0, 714.4, "English", "PG-13", "Warner Bros.", 8.5, 3.3, "Paul Atreides unites with Chani and the Fremen while seeking revenge against the conspirators who destroyed his family."),
    ("Alien", 1979, "Horror|Sci-Fi", "Ridley Scott", 117, 11.0, 106.3, "English", "R", "20th Century Fox", 8.5, 2.5, "The crew of a commercial spacecraft encounter a deadly lifeform after investigating an unknown transmission."),
    ("Django Unchained", 2012, "Drama|Western", "Quentin Tarantino", 165, 100.0, 426.0, "English", "R", "The Weinstein Company", 8.5, 3.2, "With the help of a German bounty-hunter, a freed slave sets out to rescue his wife from a brutal plantation owner in Mississippi."),
    ("Spider-Man: Across the Spider-Verse", 2023, "Animation|Action|Adventure", "Joaquim Dos Santos", 140, 100.0, 690.9, "English", "PG", "Sony Pictures", 8.6, 2.9, "Miles Morales catapults across the Multiverse, where he encounters a team of Spider-People charged with protecting its very existence."),
    ("Spider-Man: Into the Spider-Verse", 2018, "Animation|Action|Adventure", "Bob Persichetti", 117, 90.0, 384.3, "English", "PG", "Sony Pictures", 8.4, 2.8, "Teen Miles Morales becomes the Spider-Man of his universe and must join with five spider-powered individuals from other dimensions to stop a threat for all realities."),
    ("Avengers: Endgame", 2019, "Action|Adventure|Sci-Fi", "Anthony Russo", 181, 356.0, 2799.4, "English", "PG-13", "Marvel Studios", 8.4, 3.7, "After the devastating events of Avengers: Infinity War, the universe is in ruins. With the help of remaining allies, the Avengers assemble once more."),
    ("Avengers: Infinity War", 2018, "Action|Adventure|Sci-Fi", "Anthony Russo", 149, 325.0, 2052.4, "English", "PG-13", "Marvel Studios", 8.4, 3.4, "The Avengers and their allies must be willing to sacrifice all in an attempt to defeat the powerful Thanos before his blitz of devastation puts an end to the universe."),
    ("Coco", 2017, "Animation|Adventure|Family", "Lee Unkrich", 105, 175.0, 814.3, "English", "PG", "Pixar Animation Studios", 8.4, 2.6, "Aspiring musician Miguel, confronted with his family's ancestral ban on music, enters the Land of the Dead to find his great-great-grandfather, a legendary singer."),
    ("Memento", 2000, "Mystery|Thriller", "Christopher Nolan", 113, 9.0, 40.0, "English", "R", "Newmarket Films", 8.4, 2.6, "A man with short-term memory loss attempts to track down his wife's murderer."),
    ("The Shining", 1980, "Drama|Horror", "Stanley Kubrick", 146, 19.0, 47.3, "English", "R", "Warner Bros.", 8.4, 2.7, "A family heads to an isolated hotel for the winter where a sinister presence influences the father into violence, while his psychic son sees horrific forebodings."),
    ("WALL-E", 2008, "Animation|Adventure|Family", "Andrew Stanton", 98, 180.0, 521.3, "English", "G", "Pixar Animation Studios", 8.4, 2.7, "In the distant future, a small waste-collecting robot inadvertently embarks on a space journey that will ultimately decide the fate of mankind."),
    ("3 Idiots", 2009, "Comedy|Drama", "Rajkumar Hirani", 170, 6.7, 90.0, "Hindi", "PG-13", "Vinod Chopra Films", 8.4, 2.5, "Two friends are searching for their long lost companion. They revisit their college days and recall the memories of their friend who inspired them to think differently."),
    ("Your Name.", 2016, "Animation|Drama|Fantasy", "Makoto Shinkai", 106, 15.0, 382.2, "Japanese", "PG", "CoMix Wave Films", 8.4, 2.4, "Two teenagers share a profound, magical connection upon discovering they are swapping bodies."),
    ("Oldboy", 2003, "Action|Drama|Mystery", "Park Chan-wook", 120, 3.0, 17.5, "Korean", "R", "Show East", 8.3, 2.3, "After being kidnapped and imprisoned for fifteen years, Oh Dae-Su is released, only to find that he must find his captor in five days."),
    ("Oppenheimer", 2023, "Biography|Drama|History", "Christopher Nolan", 180, 100.0, 976.1, "English", "R", "Universal Pictures", 8.3, 3.4, "The story of American scientist J. Robert Oppenheimer and his role in the development of the atomic bomb."),
    ("Inglourious Basterds", 2009, "Adventure|Drama|War", "Quentin Tarantino", 153, 70.0, 321.5, "English", "R", "Universal Pictures", 8.4, 3.0, "In Nazi-occupied France during World War II, a plan to assassinate Nazi leaders by a group of Jewish U.S. soldiers coincides with a theatre owner's vengeful plans."),
    ("Joker", 2019, "Crime|Drama|Thriller", "Todd Phillips", 122, 55.0, 1078.9, "English", "R", "Warner Bros.", 8.4, 3.2, "During the 1980s, a failed stand-up comedian is driven insane and turns to a life of crime and chaos in Gotham City while becoming an infamous psychopathic crime figure."),
    ("Good Will Hunting", 1997, "Drama|Romance", "Gus Van Sant", 126, 10.0, 225.9, "English", "R", "Miramax", 8.3, 2.6, "Will Hunting, a janitor at M.I.T., has a gift for mathematics, but needs help from a psychologist to find direction in his life."),
    ("Toy Story", 1995, "Animation|Adventure|Comedy", "John Lasseter", 81, 30.0, 394.4, "English", "G", "Pixar Animation Studios", 8.3, 2.6, "A cowboy doll is profoundly threatened and jealous when a new spaceman action figure supplants him as top toy in a boy's bedroom."),
    ("Eternal Sunshine of the Spotless Mind", 2004, "Drama|Romance|Sci-Fi", "Michel Gondry", 108, 20.0, 74.0, "English", "R", "Focus Features", 8.3, 2.5, "When their relationship turns sour, a couple undergoes a medical procedure to have each other erased from their memories for ever."),
    ("Amélie", 2001, "Comedy|Romance", "Jean-Pierre Jeunet", 122, 10.0, 174.2, "French", "R", "UGC Fox Distribution", 8.3, 2.3, "Amélie is an innocent and naive girl in Paris with her own sense of justice. She decides to help those around her and, along the way, discovers love."),
    ("Top Gun: Maverick", 2022, "Action|Drama", "Joseph Kosinski", 130, 170.0, 1495.7, "English", "PG-13", "Paramount Pictures", 8.2, 3.1, "After thirty years, Maverick is still pushing the envelope as a top naval aviator, but must confront ghosts of his past when he leads TOP GUN's elite graduates."),
    ("The Wolf of Wall Street", 2013, "Biography|Comedy|Crime", "Martin Scorsese", 180, 100.0, 406.9, "English", "R", "Paramount Pictures", 8.2, 3.1, "Based on the true story of Jordan Belfort, from his rise to a wealthy stock-broker living the high life to his fall involving crime, corruption and the federal government."),
    ("Shutter Island", 2010, "Mystery|Thriller", "Martin Scorsese", 138, 80.0, 294.8, "English", "R", "Paramount Pictures", 8.2, 2.9, "Teddy Daniels and Chuck Aule, two US marshals, are sent to an asylum on a remote island in order to investigate the disappearance of a patient."),
    ("Green Book", 2018, "Biography|Comedy|Drama", "Peter Farrelly", 130, 23.0, 321.8, "English", "PG-13", "Universal Pictures", 8.2, 2.4, "A working-class Italian-American bouncer becomes the driver for an African-American classical pianist on a tour of venues through the 1960s American South."),
    ("No Country for Old Men", 2007, "Crime|Drama|Thriller", "Ethan Coen", 122, 25.0, 171.6, "English", "R", "Miramax", 8.2, 2.6, "Violence and mayhem ensue after a hunter stumbles upon a drug deal gone wrong and more than two million dollars in cash near the Rio Grande."),
    ("Pan's Labyrinth", 2006, "Drama|Fantasy|War", "Guillermo del Toro", 118, 19.0, 83.9, "Spanish", "R", "Warner Bros.", 8.2, 2.3, "In the Falangist Spain of 1944, the bookish young stepdaughter of a sadistic army officer escapes into an eerie but captivating fantasy world."),
    ("Up", 2009, "Animation|Adventure|Comedy", "Pete Docter", 96, 175.0, 735.1, "English", "PG", "Pixar Animation Studios", 8.3, 2.6, "78-year-old Carl Fredricksen travels to Paradise Falls in his house equipped with balloons, inadvertently taking a young stowaway."),
    ("The Truman Show", 1998, "Comedy|Drama", "Peter Weir", 103, 60.0, 264.1, "English", "PG", "Paramount Pictures", 8.2, 2.6, "An insurance salesman discovers his whole life is actually a reality TV show."),
    ("Jurassic Park", 1993, "Action|Adventure|Sci-Fi", "Steven Spielberg", 127, 63.0, 1114.4, "English", "PG-13", "Universal Pictures", 8.2, 2.9, "A pragmatic paleontologist touring an almost complete theme park on an island in Central America is tasked with protecting a couple of kids after a power failure."),
    ("Finding Nemo", 2003, "Animation|Adventure|Comedy", "Andrew Stanton", 100, 94.0, 941.6, "English", "G", "Pixar Animation Studios", 8.2, 2.6, "After his son is captured in the Great Barrier Reef and taken to Sydney, a timid clownfish sets out on a journey to bring him home."),
    ("Kill Bill: Vol. 1", 2003, "Action|Crime|Thriller", "Quentin Tarantino", 111, 30.0, 180.9, "English", "R", "Miramax", 8.2, 2.6, "After awakening from a four-year coma, a former assassin wreaks vengeance on the team of assassins who betrayed her."),
    ("Inside Out", 2015, "Animation|Adventure|Comedy", "Pete Docter", 95, 175.0, 858.8, "English", "PG", "Pixar Animation Studios", 8.1, 2.5, "After young Riley is uprooted from her Midwest life and moved to San Francisco, her emotions - Joy, Fear, Anger, Disgust and Sadness - conflict on how best to navigate a new city."),
    ("Mad Max: Fury Road", 2015, "Action|Adventure|Sci-Fi", "George Miller", 120, 150.0, 380.4, "English", "R", "Warner Bros.", 8.1, 2.8, "In a post-apocalyptic wasteland, a woman rebels against a tyrannical ruler in search for her homeland with the aid of a group of female prisoners and a drifter named Max."),
    ("Gone Girl", 2014, "Drama|Mystery|Thriller", "David Fincher", 149, 61.0, 369.3, "English", "R", "20th Century Fox", 8.1, 2.6, "With his wife's disappearance having become the focus of an intense media circus, a man sees the spotlight turned on him when it's suspected that he may not be innocent."),
    ("Prisoners", 2013, "Crime|Drama|Mystery", "Denis Villeneuve", 153, 46.0, 122.1, "English", "R", "Warner Bros.", 8.1, 2.5, "When Keller Dover's daughter and her friend go missing, he takes matters into his own hands as the police pursue multiple leads and the pressure mounts."),
    ("The Grand Budapest Hotel", 2014, "Adventure|Comedy|Crime", "Wes Anderson", 99, 25.0, 173.0, "English", "R", "Fox Searchlight Pictures", 8.1, 2.5, "A writer encounters the owner of an aging high-class hotel, who tells him of his early years serving as a lobby boy in the hotel's glorious years under an exceptional concierge."),
    ("Ford v Ferrari", 2019, "Action|Biography|Drama", "James Mangold", 152, 97.6, 225.5, "English", "PG-13", "20th Century Fox", 8.1, 2.3, "American car designer Carroll Shelby and driver Ken Miles battle corporate interference and the laws of physics to build a revolutionary race car for Ford in order to defeat Ferrari."),
    ("Logan", 2017, "Action|Drama|Sci-Fi", "James Mangold", 137, 97.0, 619.2, "English", "R", "20th Century Fox", 8.1, 2.6, "In a future where mutants are nearly extinct, an elderly and weary Logan leads a quiet life. But when Laura, a mutant child pursued by scientists, comes to him for help, he must get her to safety."),
    ("Blade Runner 2049", 2017, "Action|Drama|Sci-Fi", "Denis Villeneuve", 164, 150.0, 267.7, "English", "R", "Warner Bros.", 8.0, 2.6, "Young Blade Runner K's discovery of a long-buried secret leads him to track down former Blade Runner Rick Deckard, who's been missing for thirty years."),
    ("Dune", 2021, "Action|Adventure|Sci-Fi", "Denis Villeneuve", 155, 165.0, 434.8, "English", "PG-13", "Warner Bros.", 8.0, 2.8, "A noble family becomes embroiled in a war for control over the galaxy's most valuable asset while its heir becomes troubled by visions of a dark future."),
    ("La La Land", 2016, "Comedy|Drama|Music", "Damien Chazelle", 128, 30.0, 472.0, "English", "PG-13", "Lionsgate", 8.0, 2.7, "While navigating their careers in Los Angeles, a pianist and an actress fall in love while attempting to reconcile their aspirations for the future."),
    ("The Martian", 2015, "Adventure|Drama|Sci-Fi", "Ridley Scott", 144, 108.0, 630.6, "English", "PG-13", "20th Century Fox", 8.0, 2.7, "An astronaut becomes stranded on Mars after his team assume him dead, and must rely on his ingenuity to find a way to signal to Earth that he is alive."),
    ("Pirates of the Caribbean: The Curse of the Black Pearl", 2003, "Action|Adventure|Fantasy", "Gore Verbinski", 143, 140.0, 654.3, "English", "PG-13", "Walt Disney Pictures", 8.1, 2.7, "Blacksmith Will Turner teams up with eccentric pirate 'Captain' Jack Sparrow to save his love, the governor's daughter, from Jack's former pirate allies, who are now undead."),
    ("Ratatouille", 2007, "Animation|Adventure|Comedy", "Brad Bird", 111, 150.0, 623.7, "English", "G", "Pixar Animation Studios", 8.1, 2.5, "A rat who can cook makes an unusual alliance with a young kitchen worker at a famous Paris restaurant."),
    ("Deadpool", 2016, "Action|Comedy", "Tim Miller", 108, 58.0, 782.8, "English", "R", "20th Century Fox", 8.0, 2.8, "A wisecracking mercenary gets experimented on and becomes immortal yet hideously scarred, and sets out to track down the man who ruined his looks."),
    ("Arrival", 2016, "Drama|Mystery|Sci-Fi", "Denis Villeneuve", 116, 47.0, 203.4, "English", "PG-13", "Paramount Pictures", 7.9, 2.5, "A linguist works with the military to communicate with alien lifeforms after twelve mysterious spacecraft appear around the world."),
    ("Knives Out", 2019, "Comedy|Crime|Mystery", "Rian Johnson", 130, 40.0, 312.9, "English", "PG-13", "Lionsgate", 7.9, 2.5, "A detective investigates the death of a patriarch of an eccentric, combative family."),
    ("Iron Man", 2008, "Action|Adventure|Sci-Fi", "Jon Favreau", 126, 140.0, 585.8, "English", "PG-13", "Marvel Studios", 7.9, 2.8, "After being held captive in an Afghan cave, billionaire engineer Tony Stark creates a unique weaponized suit of armor to fight evil."),
    ("Avatar", 2009, "Action|Adventure|Fantasy", "James Cameron", 162, 237.0, 2923.7, "English", "PG-13", "20th Century Fox", 7.9, 3.1, "A paraplegic Marine dispatched to the moon Pandora on a unique mission becomes torn between following his orders and protecting the world he feels is his home."),
    ("Titanic", 1997, "Drama|Romance", "James Cameron", 194, 200.0, 2264.7, "English", "PG-13", "Paramount Pictures", 7.9, 3.2, "A seventeen-year-old aristocrat falls in love with a kind but poor artist aboard the luxurious, ill-fated R.M.S. Titanic."),
    ("Everything Everywhere All at Once", 2022, "Action|Adventure|Comedy", "Daniel Kwan", 139, 25.0, 143.4, "English", "R", "A24", 7.8, 2.6, "A middle-aged Chinese immigrant is swept up into an insane adventure in which she alone can save existence by exploring other universes and connecting with the lives she could have led."),
    ("The Batman", 2022, "Action|Crime|Drama", "Matt Reeves", 176, 200.0, 772.2, "English", "PG-13", "Warner Bros.", 7.8, 2.7, "When a sadistic serial killer begins murdering key political figures in Gotham, Batman is forced to investigate the city's hidden corruption and question his family's involvement."),
    ("The Social Network", 2010, "Biography|Drama", "David Fincher", 120, 40.0, 224.9, "English", "PG-13", "Sony Pictures", 7.8, 2.4, "As Harvard student Mark Zuckerberg creates the social networking site that would become known as Facebook, he is sued by the twins who claimed he stole their idea."),
    ("Get Out", 2017, "Horror|Mystery|Thriller", "Jordan Peele", 104, 4.5, 255.7, "English", "R", "Universal Pictures", 7.8, 2.4, "A young African-American visits his white girlfriend's parents for the weekend, where his simmering uneasiness about their reception of him eventually reaches a boiling point."),
    ("Skyfall", 2012, "Action|Adventure|Thriller", "Sam Mendes", 143, 200.0, 1108.6, "English", "PG-13", "Sony Pictures", 7.8, 2.5, "James Bond's loyalty to M is tested when her past comes back to haunt her. When MI6 comes under attack, 007 must track down and destroy the threat, no matter how personal the cost."),
    ("Poor Things", 2023, "Comedy|Drama|Romance", "Yorgos Lanthimos", 141, 35.0, 117.6, "English", "R", "Searchlight Pictures", 7.8, 2.0, "The incredible tale about the fantastical evolution of Bella Baxter, a young woman brought back to life by the brilliant and unorthodox scientist Dr. Godwin Baxter."),
    ("Past Lives", 2023, "Drama|Romance", "Celine Song", 106, 12.0, 42.6, "Korean", "PG-13", "A24", 7.8, 1.9, "Nora and Hae Sung, two deeply connected childhood friends, are wrested apart after Nora's family emigrates from South Korea. Twenty years later, they are reunited for one fateful week."),
    ("Gravity", 2013, "Drama|Sci-Fi|Thriller", "Alfonso Cuarón", 91, 100.0, 723.2, "English", "PG-13", "Warner Bros.", 7.7, 2.4, "Dr. Ryan Stone, an engineer on her first space mission, and Matt Kowalski, an astronaut on his final expedition, have to survive in space after they are hit by debris."),
    ("Roma", 2018, "Drama", "Alfonso Cuarón", 135, 15.0, 5.1, "Spanish", "R", "Netflix", 7.7, 1.8, "A year in the life of a middle-class family's maid in Mexico City in the early 1970s."),
    ("Baby Driver", 2017, "Action|Crime|Drama", "Edgar Wright", 113, 34.0, 226.9, "English", "R", "Sony Pictures", 7.5, 2.2, "After being coerced into working for a crime boss, a young getaway driver finds himself taking part in a heist doomed to fail."),
    ("Black Panther", 2018, "Action|Adventure|Sci-Fi", "Ryan Coogler", 134, 200.0, 1349.9, "English", "PG-13", "Marvel Studios", 7.3, 2.6, "T'Challa, heir to the hidden but advanced kingdom of Wakanda, must step forward to lead his people into a new future and must confront a challenger from his country's past."),
    ("A Quiet Place", 2018, "Drama|Horror|Sci-Fi", "John Krasinski", 90, 17.0, 341.0, "English", "PG-13", "Paramount Pictures", 7.5, 2.3, "A family struggles for survival in a world where most humans have been killed by blind but noise-sensitive creatures. They are forced to communicate in sign language to keep the creatures at bay."),
    ("Avatar: The Way of Water", 2022, "Action|Adventure|Fantasy", "James Cameron", 192, 350.0, 2320.2, "English", "PG-13", "20th Century Studios", 7.5, 2.5, "Jake Sully lives with his newfound family formed on the extrasolar moon Pandora. Once a familiar threat returns to finish what was previously started, Jake must work with Neytiri."),
    ("Inside Out 2", 2024, "Animation|Adventure|Comedy", "Kelsey Mann", 96, 200.0, 1698.0, "English", "PG", "Pixar Animation Studios", 7.6, 2.5, "A sequel that features Riley entering puberty and experiencing brand new, more complex emotions as a result."),
    ("Deadpool & Wolverine", 2024, "Action|Adventure|Comedy", "Shawn Levy", 128, 200.0, 1338.0, "English", "R", "Marvel Studios", 7.6, 2.6, "Deadpool is offered a place in the Marvel Cinematic Universe by the Time Variance Authority, but instead recruits a variant of Wolverine to save his universe from extinction."),
    ("Tenet", 2020, "Action|Sci-Fi|Thriller", "Christopher Nolan", 150, 205.0, 365.3, "English", "PG-13", "Warner Bros.", 7.3, 2.3, "Armed with only one word, Tenet, and fighting for the survival of the entire world, a Protagonist journeys through a twilight world of international espionage on a mission that will unfold in something beyond real time."),
    ("Barbie", 2023, "Adventure|Comedy|Fantasy", "Greta Gerwig", 114, 145.0, 1445.6, "English", "PG-13", "Warner Bros.", 6.8, 2.6, "Barbie and Ken are having the time of their lives in the colorful and seemingly perfect world of Barbie Land. However, when they get a chance to go to the real world, they soon discover the joys and perils of living among humans."),
    ("Wonder Woman", 2017, "Action|Adventure|Fantasy", "Patty Jenkins", 141, 149.0, 823.9, "English", "PG-13", "Warner Bros.", 7.3, 2.3, "When a pilot crashes and tells of conflict in the outside world, Diana, an Amazonian warrior in training, leaves home to fight a war, discovering her full powers and true destiny."),
    ("Jaws", 1975, "Adventure|Mystery|Thriller", "Steven Spielberg", 124, 7.0, 476.5, "English", "PG", "Universal Pictures", 8.1, 2.5, "When a killer shark unleashes chaos on a beach community off Cape Cod, it's up to a local sheriff, a marine biologist, and an old seafarer to hunt the beast down."),
    ("Back to the Future", 1985, "Adventure|Comedy|Sci-Fi", "Robert Zemeckis", 116, 19.0, 388.8, "English", "PG", "Universal Pictures", 8.5, 3.0, "Marty McFly, a 17-year-old high school student, is accidentally sent 30 years into the past in a time-traveling DeLorean invented by his close friend, the maverick scientist Doc Brown."),
    ("Blade Runner", 1982, "Action|Drama|Sci-Fi", "Ridley Scott", 117, 28.0, 41.8, "English", "R", "Warner Bros.", 8.1, 2.5, "A blade runner must pursue and terminate four replicants who stole a ship in space and have returned to Earth to find their creator."),
    ("2001: A Space Odyssey", 1968, "Adventure|Sci-Fi", "Stanley Kubrick", 149, 12.0, 146.0, "English", "G", "Metro-Goldwyn-Mayer", 8.3, 2.5, "After uncovering a mysterious artifact buried beneath the Lunar surface, a spacecraft is sent to Jupiter to find its origins: a spacecraft manned by two men and the supercomputer HAL 9000."),
    ("Heat", 1995, "Action|Crime|Drama", "Michael Mann", 170, 60.0, 187.4, "English", "R", "Warner Bros.", 8.3, 2.5, "A group of high-end professional thieves start to feel the heat from the LAPD when they unknowingly leave a structural clue at their latest heist."),
    ("Casino Royale", 2006, "Action|Adventure|Thriller", "Martin Campbell", 144, 150.0, 616.5, "English", "PG-13", "Sony Pictures", 8.0, 2.5, "After earning 00 status and a licence to kill, secret agent James Bond sets out on his first mission as 007. Bond must defeat a private banker funding terrorists in a high-stakes game of poker at Casino Royale."),
    ("Her", 2013, "Drama|Romance|Sci-Fi", "Spike Jonze", 126, 23.0, 48.5, "English", "R", "Warner Bros.", 8.0, 2.3, "In a near future, a lonely writer develops an unlikely relationship with an operating system designed to meet his every need."),
    ("Ex Machina", 2014, "Drama|Sci-Fi|Thriller", "Alex Garland", 108, 15.0, 38.3, "English", "R", "A24", 7.7, 2.3, "A young programmer is selected to participate in a ground-breaking experiment in synthetic intelligence by evaluating the human qualities of a highly advanced humanoid A.I."),
    ("Midsommar", 2019, "Drama|Horror|Mystery", "Ari Aster", 148, 9.0, 48.1, "English", "R", "A24", 7.1, 2.1, "A couple travels to Northern Europe to visit a rural hometown's fabled Swedish mid-summer festival. What begins as an idyllic retreat quickly devolves into an increasingly violent and bizarre competition."),
    ("The Meg", 2018, "Action|Horror|Sci-Fi", "Jon Turteltaub", 113, 130.0, 530.5, "English", "PG-13", "Warner Bros.", 5.7, 1.8, "A group of scientists exploring the Marianas Trench encounter the largest marine predator that has ever existed - the Megalodon."),
    ("Venom", 2018, "Action|Adventure|Sci-Fi", "Ruben Fleischer", 112, 100.0, 856.1, "English", "PG-13", "Sony Pictures", 6.6, 2.2, "A failed reporter is bonded to an alien entity, one of many symbiotes who have invaded Earth. But the being takes a liking to Earth and decides to protect it."),
    ("Morbius", 2022, "Action|Adventure|Horror", "Daniel Espinosa", 104, 75.0, 167.5, "English", "PG-13", "Sony Pictures", 5.1, 1.6, "Biochemist Michael Morbius tries to cure himself of a rare blood disease, but he inadvertently infects himself with a form of vampirism instead."),
    ("Madame Web", 2024, "Action|Adventure|Sci-Fi", "S.J. Clarkson", 116, 80.0, 100.5, "English", "PG-13", "Sony Pictures", 4.0, 1.5, "Cassandra Webb is a New York metropolis paramedic who begins to demonstrate signs of clairvoyance. Forced to challenge revelations about her past, she needs to safeguard three young women from a deadly adversary.")
]


def generate_raw_datasets():
    os.makedirs(RAW_DIR, exist_ok=True)
    movies_rows = []
    movie_id_counter = 1001

    inherent_quality = {}
    popularity_weight = {}

    for (
        title, year, genres, director, runtime, budget, revenue,
        lang, cert, studio, true_imdb, pop_w, synopsis
    ) in REAL_MOVIES:
        mid = f"MOV-{movie_id_counter}"
        movie_id_counter += 1
        movies_rows.append({
            "movie_id": mid,
            "title": title,
            "release_year": year,
            "genres": genres,
            "director": director,
            "runtime_min": runtime,
            "budget_musd": budget,
            "box_office_musd": revenue,
            "language": lang,
            "content_rating": cert,
            "synopsis": synopsis,
            "production_studio": studio
        })
        inherent_quality[mid] = true_imdb
        popularity_weight[mid] = pop_w

    canonical_movies = list(movies_rows)

    # 2. Inject intentional data quality issues into raw_movie_details.csv
    # 2a. Inject 18 duplicate real-life title entries (different movie_id, messy casing/spaces, missing fields)
    duplicate_IDs_map = {}  # dup_movie_id -> canonical_movie_id
    dup_candidates = random.sample(canonical_movies, 18)
    for orig in dup_candidates:
        dup_id = f"MOV-{movie_id_counter}"
        movie_id_counter += 1
        duplicate_IDs_map[dup_id] = orig["movie_id"]

        variant_type = random.choice(["upper", "lower", "spaces", "exact"])
        if variant_type == "upper":
            messy_title = orig["title"].upper()
        elif variant_type == "lower":
            messy_title = f"  {orig['title'].lower()} "
        elif variant_type == "spaces":
            messy_title = f" {orig['title']}   "
        else:
            messy_title = orig["title"]

        movies_rows.append({
            "movie_id": dup_id,
            "title": messy_title,
            "release_year": orig["release_year"],
            "genres": orig["genres"] if random.random() > 0.35 else np.nan,
            "director": orig["director"] if random.random() > 0.4 else np.nan,
            "runtime_min": orig["runtime_min"] if random.random() > 0.5 else np.nan,
            "budget_musd": np.nan if random.random() > 0.4 else orig["budget_musd"],
            "box_office_musd": orig["box_office_musd"] if random.random() > 0.5 else np.nan,
            "language": orig["language"],
            "content_rating": orig["content_rating"] if random.random() > 0.35 else np.nan,
            "synopsis": orig["synopsis"],
            "production_studio": orig["production_studio"]
        })
        inherent_quality[dup_id] = inherent_quality[orig["movie_id"]]
        popularity_weight[dup_id] = popularity_weight[orig["movie_id"]] * 0.35

    # 2b. Inject missing values into ~18 canonical movie records so ETL coalescing & imputation are demonstrated
    for idx in random.sample(range(len(canonical_movies)), 18):
        field_to_null = random.choice(["budget_musd", "runtime_min", "director", "box_office_musd", "content_rating", "genres"])
        movies_rows[idx][field_to_null] = np.nan

    random.shuffle(movies_rows)
    df_raw_movies = pd.DataFrame(movies_rows)
    raw_movies_path = os.path.join(RAW_DIR, "raw_movie_details.csv")
    df_raw_movies.to_csv(raw_movies_path, index=False)

    # 3. Generate raw_movie_ratings.csv (7,800+ individual audience ratings calibrated to true IMDb ratings)
    ratings_rows = []
    rating_id_counter = 50001
    sources = ["IMDb Verified", "Letterboxd", "RottenTomatoes Audience", "Metacritic Users", "TMDB Community"]
    age_groups = ["18-24", "25-34", "35-44", "45-54", "55+"]
    base_date = datetime(2023, 1, 1)

    all_movie_ids = [m["movie_id"] for m in movies_rows]
    weights = np.array([popularity_weight[mid] ** 1.65 for mid in all_movie_ids])
    weights = weights / weights.sum()

    num_ratings = 7800
    chosen_mids = np.random.choice(all_movie_ids, size=num_ratings, p=weights)

    for mid in chosen_mids:
        mean_score = inherent_quality[mid]
        # Tight standard deviation around true IMDb score for realistic accuracy
        raw_score = np.random.normal(mean_score, 0.72)
        score = round(float(np.clip(raw_score, 1.0, 10.0)), 1)

        # Occasional missing rating value (~1.2%)
        if random.random() < 0.012:
            score = np.nan

        user_id = f"USR-{random.randint(100, 1850)}"
        days_offset = random.randint(0, 1020)
        ts = (base_date + timedelta(days=days_offset, hours=random.randint(0, 23), minutes=random.randint(0, 59))).strftime("%Y-%m-%d %H:%M:%S")

        ratings_rows.append({
            "rating_id": f"RAT-{rating_id_counter}",
            "movie_id": mid,
            "user_id": user_id,
            "rating": score,
            "review_Platform": random.choice(sources),
            "viewer_age_group": random.choice(age_groups),
            "is_verified_viewer": random.choice([True, True, True, False]),
            "helpful_votes": int(np.random.exponential(15)),
            "rated_at": ts
        })
        rating_id_counter += 1

    # 3b. Inject 140 duplicate user-movie rating rows
    sample_for_dups = random.sample(ratings_rows[:2500], 140)
    for r in sample_for_dups:
        dup_r = dict(r)
        dup_r["rating_id"] = f"RAT-{rating_id_counter}"
        rating_id_counter += 1
        ratings_rows.append(dup_r)

    # 3c. Inject 35 orphan ratings (movie_id that doesn't exist in movie_details)
    for _ in range(35):
        ratings_rows.append({
            "rating_id": f"RAT-{rating_id_counter}",
            "movie_id": f"MOV-999{random.randint(1, 9)}",
            "user_id": f"USR-{random.randint(100, 900)}",
            "rating": round(random.uniform(4.0, 9.0), 1),
            "review_Platform": random.choice(sources),
            "viewer_age_group": random.choice(age_groups),
            "is_verified_viewer": False,
            "helpful_votes": 0,
            "rated_at": "2025-05-10 12:00:00"
        })
        rating_id_counter += 1

    # 3d. Inject 22 out-of-bounds ratings (e.g., 12.0 or -1.0)
    for idx in random.sample(range(len(ratings_rows)), 22):
        ratings_rows[idx]["rating"] = random.choice([11.5, 14.0, -2.0, 0.0])

    random.shuffle(ratings_rows)
    df_raw_ratings = pd.DataFrame(ratings_rows)
    raw_ratings_path = os.path.join(RAW_DIR, "raw_movie_ratings.csv")
    df_raw_ratings.to_csv(raw_ratings_path, index=False)

    return {
        "raw_movies_path": raw_movies_path,
        "raw_ratings_path": raw_ratings_path,
        "raw_movies_count": len(df_raw_movies),
        "raw_ratings_count": len(df_raw_ratings)
    }


if __name__ == "__main__":
    info = generate_raw_datasets()
    print(f"Generated real-life movie raw datasets: {info}")
