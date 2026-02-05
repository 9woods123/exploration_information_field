from eif_map import *



def main():

    eif_table, sdf_field, map2d= map_generate()
    plot_eif_and_sdf(eif_table, sdf_field,eif_table.Yaw,map2d ,show_sdf=False)



if __name__ == "__main__":
    main()
