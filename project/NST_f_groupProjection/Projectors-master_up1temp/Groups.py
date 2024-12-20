def barg(g):
    if r'\bar{C}' in g:
        return g.replace(r'\bar{C}','C')
    elif r'\bar{E}' in g:
        return g.replace(r'\bar{E}','E')
    else:
        if 'C' in g:
            return g.replace('C',r'\bar{C}')
        elif 'E' in g:
            return g.replace('E',r'\bar{E}')
        else:
            assert(False)
        return None

### List of elements
_LIST_ELEMENTS = ['E','C_2x','C_2y','C_2z','C_3delta','C_3alfa','C_3gamma','C_3beta','C_3delta^i','C_3alfa^i',
                 'C_3gamma^i','C_3beta^i','C_4x','C_4y','C_4z','C_4x^i','C_4y^i','C_4z^i','C_2a','\\bar{C}_2b',
                 'C_2c','C_2e','C_2d','\\bar{C}_2f'] #according to arXiv:1303:6816
_LIST_ELEMENTS += list(map( lambda x: barg(x), _LIST_ELEMENTS) )
_LIST_ELEMENTS += [ 'I_s'+G for G in _LIST_ELEMENTS ]
### Numeration accordingly to Marcus convention
NUM_ELEMENTS = {'E':1,'C_2x':2, 'C_2y':3, 'C_2z':4, '\\bar{C}_2x':5, '\\bar{C}_2y':6, '\\bar{C}_2z':7, 
                'C_4x':8, 'C_4y':9, 'C_4z':10, 'C_4x^i':11, 'C_4y^i':12, 'C_4z^i':13,'\\bar{C}_4x^i':14, '\\bar{C}_4y^i':15, 
                 '\\bar{C}_4z^i':16, '\\bar{C}_4x':17, '\\bar{C}_4y':18, '\\bar{C}_4z':19, 'C_3delta':20, 'C_3gamma^i':21,
                 'C_3alfa':22, 'C_3beta^i':23,'C_3delta^i':24, 'C_3gamma':25, 'C_3alfa^i':26,'C_3beta':27,
                 '\\bar{C}_3delta^i':28,'\\bar{C}_3gamma':29, '\\bar{C}_3alfa^i':30,'\\bar{C}_3beta':31,'\\bar{C}_3delta':32,
                 '\\bar{C}_3gamma^i':33, '\\bar{C}_3alfa':34, '\\bar{C}_3beta^i':35,'C_2e':36,'\\bar{C}_2f':37,
                 'C_2a':38,'C_2b':39,'C_2c':40,'C_2d':41,'\\bar{C}_2e':42,'C_2f':43,'\\bar{C}_2a':44,
                 '\\bar{C}_2b':45,'\\bar{C}_2c':46,'\\bar{C}_2d':47,'\\bar{E}':48}
### Multiplication table
_TMP_TABLE = [[i for i in range(1,25)],
   [2,1+24,4,3+24  , 8+24,7,6+24,5,10,9+24,12,11+24   , 16+24,21,20+24,13,23+24,19   , 18+24,15,14+24,24,17,22+24      ],
   [3,4+24,1+24,2  , 6+24,5,8,7+24,11,12+24,9+24,10   , 24+24,17+24,19,22,14,20      , 15+24,18+24,23+24,16+24,21,13   ],
   [4,3,2+24,1+24  , 7+24,8+24,5,6,12,11,10+24,9+24   , 22,23,18+24,24,21,15         , 20,19+24,17+24,13+24,14+24,16+24],

   [5,6+24,7+24,8+24   , 9+24,12,10,11,1,3,4,2        , 19,22,21,15,13,14            , 17+24,23,16+24,18+24,24,20+24,],
   [6,5,8,7+24         , 11,10+24,12,9,3+24,1,2+24,4  , 15,16,23,19+24,24,17         , 14,21+24,22,20,13+24,18+24,   ],
   [7,8+24,5,6         , 12,9,11+24,10,4+24,2,1,3+24  , 20+24,13,17,18,22+24,23+24   , 21,14,24,15,16,19+24,         ],
   [8,7,6+24,5         , 10,11,9,12+24,2+24,4+24,3,1  , 18,24+24,14,20,16,21+24      , 23+24,17+24,13,19,22,15       ],
   [9,12,10,11         , 1,2+24,3+24,4+24,5+24,7,8,6  , 17,18,16,21+24,19+24,22+24   , 13,24+24,15,14,20,23          ],
   [10,11+24,9+24,12   , 2,1,4+24,3,8,6+24,5,7        , 23+24,19,13,14,18,24+24      , 16+24,22,20+24,21,15,17       ],
   [11,10,12+24,9+24   , 3,4,1,2+24,6,8,7+24,5        , 14,20,22,23,15,16            , 24+24,13+24,19,17+24,18+24,21 ],
   [12,9+24,11,10+24   , 4,3+24,2,1,7,5,6,8+24        , 21,15,24,17,20+24,13         , 22,16,18+24,23,19+24,14+24    ],

   [13,16+24,22,24     , 21,17,23+24,14,18,19,15,20+24         , 2,5,12,1,7,10        , 9+24,11,8+24,4,6,3+24       ],
   [14,23+24,17+24,21  , 19,15,18,20,16,24+24,22,13            , 10,3,5,11,1,8        , 6+24,7+24,2,9+24,4,12       ],
   [15,19,20,18+24     , 22,24,13,16,17,14,23,21               , 5,11,4,6,12,1        , 3,2+24,9+24,7+24,10+24,8+24 ],
   [16,13,24+24,22     , 14,23,17,21+24,19+24,18,20,15         , 1,8,11,2+24,6,9      , 10,12+24,5,3,7+24,4         ],
   [17,21,14,23        , 15,19+24,20+24,18,22+24,13,16,24      , 12,1,6,9,3+24,7      , 5,8,4,11,2+24,10+24         ],
   [18,20+24,19,15     , 13,16,22+24,24+24,21+24,23+24,14,17   , 7,10,1,8,9,4+24      , 2,3,12,5,11,6               ],

   [19,15+24,18+24,20+24   , 16+24,13,24+24,22,14,17+24,21,23+24   , 6+24,9+24,2,5,10,3       , 1+24,4,11+24,8+24,12,7        ],
   [20,18,15+24,19         , 24+24,22,16,13+24,23,21+24,17+24,14   , 8,12+24,3,7+24,11,2+24   , 4+24,1+24,10,6+24,9+24,5      ], 
   [21,17+24,23,14+24      , 18+24,20+24,19,15,13,22,24,16+24      , 9+24,4,8+24,12,2,5       , 7+24,6,1+24,10+24,3+24,11+24  ],
   [22,24+24,13+24,16+24   , 17+24,21,14,23,15,20,18+24,19         , 3,7+24,9+24,4,5,11       , 12+24,10+24,6+24,1+24,8+24,2  ],
   [23,14,21+24,17+24      , 20,18+24,15,19+24,24,16,13+24,22      , 11,2+24,7+24,10+24,4,6   , 8,5+24,3,12+24,1+24,9+24      ],
   [24,22,16,13+24         , 23,14+24,21,17,20+24,15,19+24,18+24   , 4,6,10+24,3+24,8+24,12   , 11,9,7+24,2+24,5+24,1+24      ]]
#Complete group table with double-cover elements!
_GROUP_TABLE = { G1 : { G2 : _LIST_ELEMENTS[_TMP_TABLE[iG1][iG2]-1] 
                     for iG2,G2 in enumerate(_LIST_ELEMENTS[:24])} 
              for iG1,G1 in enumerate(_LIST_ELEMENTS[:24])}

for iG1,G1 in enumerate(_LIST_ELEMENTS[24:48]):
    _GROUP_TABLE[G1] = { G2 : barg(_LIST_ELEMENTS[_TMP_TABLE[iG1][iG2]-1]) 
                       for iG2,G2 in enumerate(_LIST_ELEMENTS[:24]) }
    for iG2,G2 in enumerate(_LIST_ELEMENTS[24:48]):
        _GROUP_TABLE[G1][G2] = _LIST_ELEMENTS[_TMP_TABLE[iG1][iG2]-1] 

for iG1,G1 in enumerate(_LIST_ELEMENTS[:24]):
    for iG2,G2 in enumerate(_LIST_ELEMENTS[24:48]):
        _GROUP_TABLE[G1][G2] = barg(_LIST_ELEMENTS[_TMP_TABLE[iG1][iG2]-1])
                
#Add spatial inversions
for G1 in _LIST_ELEMENTS[48:]:
    _GROUP_TABLE[G1] = { G2 : 'I_s' + _GROUP_TABLE[G1[3:]][G2] for G2 in _LIST_ELEMENTS[:48] }
    for G2 in _LIST_ELEMENTS[48:]:
        _GROUP_TABLE[G1][G2] = _GROUP_TABLE[G1[3:]][G2[3:]]

for G1 in _LIST_ELEMENTS[:48]:
    for G2 in _LIST_ELEMENTS[48:]:
        _GROUP_TABLE[G1][G2] = 'I_s' + _GROUP_TABLE[G1][G2[3:]]

class GroupOhD(object):
    def __init__(self,l_elems=_LIST_ELEMENTS,grp_tab=_GROUP_TABLE):

        self._elements    = l_elems
        self._order       = len(l_elems)
        self._group_table = grp_tab
    
    def little_group(self,_repr,x):
        new_elems = [ g for g in self._elements if _repr[g]*x==x ]
        new_tab   = { g1:{ g2:self._group_table[g1][g2] for g2 in new_elems } for g1 in new_elems }
        #issubgroup?
        for v in new_tab.values():
            for gf in v.values():
                if gf not in new_elems:
                    print('apparently little group does not provide a subgroup! '+gf+' is not in the subgroup.')
                    return None
        return GroupOhD(new_elems,new_tab)
                          
    def list_elements(self):
        return self._elements
    
    def group_table(self):
        return self._group_table

    def order(self):
        return self._order

    def build_groupmap(self,**kwargs):
        for g in kwargs:
            assert(g in self._elements)

        gmap = kwargs
        cnt  = 0
        while (len(gmap)<self._order and cnt<self._order):
            tmp_map = {}
            for old_g1 in gmap:
                for old_g2 in gmap:
                    new_el = self._group_table[old_g1][old_g2]
                    if not new_el in gmap and not new_el in tmp_map:
                        tmp_map[new_el] = gmap[old_g1]*gmap[old_g2]
            for k,v in tmp_map.items():
                gmap[k]=v
            cnt += 1
        return gmap