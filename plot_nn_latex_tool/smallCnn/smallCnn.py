import sys
sys.path.append('../')
from pycore.tikzeng import *
from pycore.blocks  import *

arch = [ 
    to_head('.'), 
    to_cor(),
    to_begin(),
    
    #input
    to_input( './examples/example1.png',  width=21, height=8.4),

    #block 1
    to_ConvRelu( name='convb1_1', s_filer='', n_filer=32, offset="(0,0,0)", to="(0,0,0)", width=2, height=40, depth=100 ),
    to_ConvRelu( name='convb1_2', s_filer=(99, 39), n_filer=64, offset="(0,0,0)", to="(convb1_1-east)", width=4, height=40, depth=100),
    to_Pool(name="poolb1", offset="(0,0,0)", to="(convb1_2-east)", width=4, height=10, depth=100, opacity=0.5, caption=''),
    to_BatchNorm(name="bnormb1", offset="(0,0,0)", to="(poolb1-east)", width=4, height=10, depth=100, opacity=0.5, caption=''),
    to_Dropout(name="dropoutb1", offset="(0,0,0)", to="(bnormb1-east)", width=4, height=10, depth=100, opacity=0.5, caption=''),
        
    #block 2
    to_ConvRelu( name='convb2_1', s_filer='', n_filer=64, offset="(3,0,0)", to="(dropoutb1-east)", width=4, height=10, depth=100), 
 
    to_connection( "dropoutb1", "convb2_1"),
    
    to_ConvRelu( name='convb2_2', s_filer=(99, 39), n_filer=128, offset="(0,0,0)", to="(convb2_1-east)", width=8, height=10, depth=100),
    to_Pool(name="poolb2", offset="(0,0,0)", to="(convb2_2-east)", width=8, height=3, depth=3, opacity=0.5, caption=''),
    to_BatchNorm(name="bnormb2", offset="(0,0,0)", to="(poolb2-east)", width=8, height=3, depth=3, opacity=0.5, caption='GlobalMaxPooling'),
    to_Dropout(name="dropoutb2", offset="(0,0,0)", to="(bnormb2-east)", width=8, height=3, depth=3, opacity=0.5, caption=''),
    
    #block 3
    to_Conv( name='convb3_1', s_filer='', n_filer=128, offset="(1,0,0)", to="(dropoutb2-east)", width=8, height=3, depth=3),
    to_connection( "dropoutb2", "convb3_1"),
    to_Conv( name='convb3_2', s_filer='', n_filer=35, offset="(0,0,0)", to="(convb3_1-east)", width=2, height=3, depth=3),
    
    
    # end
    to_end() 
    ]


def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()