create table population
(
    guid        integer not null
        constraint population_pk
            primary key,
    schema      varchar,
    description varchar
);

alter table population
    owner to bioexplorer;

create table glio_vascular
(
    guid                        integer not null,
    astrocyte_population_guid   integer not null,
    vasculature_population_guid integer not null,
    astrocyte_guid              integer not null,
    astrocyte_section_guid      integer not null,
    vasculature_node_guid       integer not null,
    vasculature_section_guid    integer not null,
    vasculature_segment_guid    integer not null,
    constraint glio_vascular_pk
        primary key (guid, astrocyte_population_guid, vasculature_population_guid)
);

alter table glio_vascular
    owner to bioexplorer;

create unique index glio_vascular_astrocyte_section_guid_vasculature_section_guid_v
    on glio_vascular (astrocyte_section_guid, vasculature_section_guid, vasculature_segment_guid);

