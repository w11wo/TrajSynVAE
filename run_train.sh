for city in Beijing Porto San_Francisco; do
    python Code/main.py --data_type $city
done

python Code/main.py --data_type Beijing --generate true --checkpoint RES/2026-0411-1214/0/data/Model.pth
python Code/main.py --data_type Porto --generate true --checkpoint RES/2026-0411-1213/0/data/Model.pth
python Code/main.py --data_type San_Francisco --generate true --checkpoint RES/2026-0411-0534/0/data/Model.pth

python Code/eval.py \
    --city Beijing \
    --roadmap_geo_path data/Beijing/roadmap.geo \
    --label_trajs_path RES/2026-0413-0527/0/data/labels_od.npy \
    --gen_trajs_path RES/2026-0413-0527/0/data/generated_od.npy

python Code/eval.py \
    --city Porto \
    --roadmap_geo_path data/Porto/roadmap.geo \
    --label_trajs_path RES/2026-0413-0526/0/data/labels_od.npy \
    --gen_trajs_path RES/2026-0413-0526/0/data/generated_od.npy

python Code/eval.py \
    --city San_Francisco \
    --roadmap_geo_path data/San_Francisco/roadmap.geo \
    --label_trajs_path RES/2026-0413-0512/0/data/labels_od.npy \
    --gen_trajs_path RES/2026-0413-0512/0/data/generated_od.npy