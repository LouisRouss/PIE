import dict_utilities as dict_u
import pickle

def save_obj(obj, name ):
    with open( name + '.pkl', 'wb+') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def test_reset_dict_is_reset():
    my_dico={"voiture": "véhicule à quatre roues", "vélo": "véhicule à deux roues"}
    save_obj(my_dico,"my_dico")
    dict_u.reset_dict("./my_dico.pkl")
    data_dict = dict_u.get_dict("./my_dico.pkl")
    boo=(data_dict=={})
    assert boo
    
def test_get_dict_is_correct():
    my_dico={"voiture": "véhicule à quatre roues", "vélo": "véhicule à deux roues"}
    save_obj(my_dico,"my_dico")
    data_dict = dict_u.get_dict("./my_dico.pkl")
    boo=(data_dict == my_dico)
    assert boo
    
def test_get_dict_is_not_empty():
    my_dico={"voiture": "véhicule à quatre roues", "vélo": "véhicule à deux roues"}
    save_obj(my_dico,"my_dico")
    data_dict = dict_u.get_dict("./my_dico.pkl")
    boo=(data_dict != {})
    assert boo
    