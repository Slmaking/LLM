'''
A script that extracts social network graphs based on an already character recognised corpus, in which
the character mentions are designated by char-offsets.
Attempts to visualise the desired graphs, though that isn't working fantastically yet (it's done with networkx, which
may be a bad idea)

Command to run: character_networks.py text_file character_file social_network_type protagonist delimiter

text_file: raw text of a work to extract a network from
character_file: character_file, tsv, containing at least CHARACTER, START_OFFSET, END_OFFSET, CHAPTER, PASSAGE columns
(though whether they are all necessary depends on params)
social_network_type: either 'passage' or 'conversation' where passage is a cooccurence metric (characters are within one
passage together) and conversation is self-explanatory.
protagonist: either 'all', in which case a social network is between all characters, or a name of the protagonist who's
networks we are interested in.
delimiter: 0 means uses chapters, any other number should be a percentage of the text to consider a chunk


'''
#TODO allow protagonists to be a list of names
#TODO add more validation
#TODO add other types of delimiter

import argparse
import csv
from collections import defaultdict
from itertools import permutations, combinations_with_replacement

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# -------------------------------------------
# Globals
# -------------------------------------------
IGNORE_SET = {'NON-CHARACTER', 'OTHER', '???'}


def character_to_index(all_characters):
    """
    creates an index mapping from a set of character names.
    Should probably just make a generalisable version of this function.
    :param all_characters: ideally a set, otherwise there will be weird gaps in the indexing if there are duplicates
    :return: two dictionaries - character to index and the reverse
    """
    char2idx, idx2char = {}, {}
    current_index = 0
    for char in all_characters:
        char2idx[char] = current_index
        idx2char[current_index] = char
        current_index += 1

    return char2idx, idx2char


def read_character_tsv(characterfile):
    '''
    Gets characters and other important data from a character tsv file
    :param characterfile: a tsv file, including at the very least certain headers, but in no particular order
    :return: a set of all characters and a nested dict of {chapter: passage: {'char1', 'char2'}}
    '''
    #TODO add the option to either get by chapter OR by offset
    all_characters = set()
    chapter_passage_dict = defaultdict(lambda: defaultdict(set))
    with open(characterfile) as tsvfile:
        reader = csv.DictReader(tsvfile, dialect='excel-tab')
        for row in reader:
            chapter, passage, character = int(row['CHAPTER #']), int(row['PASSAGE #']), row['CHARACTER']
            if character in IGNORE_SET:
                continue
            else:
                all_characters.add(character)
                chapter_passage_dict[chapter][passage].add(character)

    return all_characters, chapter_passage_dict


def populate_adjacency_matrix(num_characters, groupings, char2index):
    '''
    creates and populates an adjacency matrix with character interactions
    :param num_characters: an int of the number of characters, for the matrix dimensions
    :param groupings: a dict of {grouping: {character_set},...}. Technically grouping could be anything, so this is
    agnostic to the type of grouping used (based on passage, based on, anything else)
    :param char2index: a mapping of character name to index
    :return: the social network matrix based on the given dict of groupings
    '''
    new_chap_matrix = np.zeros((num_characters, num_characters))
    for passage in groupings:
        # gets all permutations of characters that occur in a particular passage. Returns an iterable of tuples.
        character_pairs = permutations(groupings[passage], 2)
        for person1, person2 in character_pairs:
            new_chap_matrix[char2index[person1]][char2index[person2]] += 1
            #TODO work out whether having matrix diagonals matters
            # update the matrix diagonals as well, just once
            # for char in passages[passage]:
            #    new_chap_matrix[char2index[char]][char2index[char]] += 1
    return new_chap_matrix


def populate_adjacency_list(num_characters, groupings, protagonist, char2index):
    '''
    Currently creates an UNWEIGHTED adjacency list, to play nicely with networkx
    :param num_characters: UNUSED
    :param groupings: a dict of {grouping: {character_set},...}. Technically grouping could be anything, so this is
    agnostic to the type of grouping used (based on passage, based on, anything else)
    :param protagonist: the person from whose perspective this network is being drawn
    :param char2index: UNUSED
    :return: a list with protagonist at index 0 and all adjacent characters at the remaining indices
    '''
    #TODO make weighted play nice with networkx or something other than networkx
    adjacent_characters, adjacency_list = set(), [protagonist]
    for passage in groupings:
        characters = groupings[passage]
        if protagonist not in characters:
            continue
        else:
            adjacent_characters = adjacent_characters | characters
    adjacency_list.extend(list(adjacent_characters))

    return adjacency_list


def extract_social_networks(chapter_passage_dict, protagonist=None, all_characters=set(), use_all_char=False):
    '''
    will be able to just add all chapters to get overall network
    :param chapter_passage_dict:
    :param all_characters if present, a list of all characters, and the matrix will be of the full character set for each 
    chapter regardless of similarity
    :return:
    '''
    #TODO THIS DOCSTRING
    #TODO document that dedup within a single passage. since want to consider one passage to be one "event"
    chapter_networks = []
    max_chapter = max(chapter_passage_dict)
    for i in range(max_chapter+1):
        passages = chapter_passage_dict[i]
        if not use_all_char:
            all_characters = set()
            for passage_char in passages.values():
                all_characters = all_characters | passage_char
        num_characters = len(all_characters)
        char2index, idx2char = character_to_index(all_characters)
        if not protagonist:
            new_chap_matrix = populate_adjacency_matrix(num_characters, passages, char2index)
            chapter_networks.append((new_chap_matrix, idx2char))
        else:
            adjlist = populate_adjacency_list(num_characters, passages, protagonist, char2index)
            chapter_networks.append(adjlist)

    return chapter_networks


def generate_network_viz(networks):
    '''
    generates a networkx graph from an adjacency matrix and saves a pyplot drawing
    :param networks: expects a list of tuples of an adjacency matrix and an index2character dict
    :return: None, saves to png files
    '''
    # TODO remove the hardcoding of test indices
    for index in range(1,5):#len(networks)):
        curr_matrix, curr_indexing = networks[index]
        curr_graph = nx.DiGraph(curr_matrix)
        #pos = nx.random_layout(curr_graph, dim=10)
        pos = nx.spring_layout(curr_graph, k=0.5, iterations=100)
        nx.draw_networkx_nodes(curr_graph, pos)
        nx.draw_networkx_edges(curr_graph, pos)
        nx.draw_networkx_labels(curr_graph, pos, curr_indexing)
        plt.axis('off')
        plt.savefig('Chapter {} network.png'.format(index))


def generate_adj_network_viz(networks):
    '''
    generates a networkx graph and from an adjacency list and saves as a pyplot drawing
    :param networks: a list of lists where the inner lists are a list of characters that are a network, the first one
    being the 'protagonist' or POV holder
    :return: None, saves to png files
    '''
    # TODO remove the hardcoding of test indices
    for index in range(1, 50, 5):
        # this is janky. Because nx expects a string
        curr_graph = nx.parse_adjlist([' '.join(networks[index])])
        nx.draw_networkx(curr_graph)
        plt.savefig('Chapter {} network.png'.format(index))



if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('filename', help='name of the file to extract social networks from')
    p.add_argument('character_file', help='a tsv with character offset data')
    p.add_argument('social_network_type', help='a string designating the type of social network. '
                                                  'options: passage or conversation')
    p.add_argument('protagonist', help='name of a perspective to use for network. "all" to use all characters.')
    p.add_argument('delimiter', type=float, help='the type of chunking to use as a timestep for dynamic social networks'
                                                 'If 0, then uses chapters, if any real number < 100, uses that '
                                                 'percentage of book as a chunk')
    args = p.parse_args()

    #get all characters and return adjacency matrix and char2idx and idx2char
    all_char, chap_passage_dict = read_character_tsv(args.character_file)

    if args.social_network_type == 'passage':
        networks = extract_social_networks(chap_passage_dict, protagonist=args.protagonist)

    #print(networks[5][0])
    #print(networks[5][1])
    #print(networks[5])
    if args.protagonist == 'all':
        generate_network_viz(networks)
    else:
        generate_adj_network_viz(networks)


    #substitute char standard names at the offset points

    #chapter split, and proceed chapter by chapter

    #sent tokenize

    #for each sentence get characters in it and increment the adjacency matrix
